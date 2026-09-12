import argparse
import cv2
import h5py
import numpy as np
import os
import os.path
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import warnings

from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from util.loss_util import AverageMeter
from util.flow_util import flow2rgb, flow_viz_np, save_checkpoint

from datasets.voxel.dataset_supervised import DatasetTestDSEC, DatasetTrainDSEC_Supervised
from models import let_flownet_voxel
from loss.loss_supervised import supervised_loss_multiscale
from loss.metrics import estimate_corresponding_gt_flow, flow_error_dense
from loss.loss_photometric import smooth_loss
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='let_flownet_voxel training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')

parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'],
                    help='solver algorithms')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--norm', default='BN',
                    help='batch norm for Transformer layers. BN: BatchNorm2d; IN: InstanceNorm2d')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--tau', type=float, default=20e-3, choices=[20e-3, 50e-3, 100e-3],
                    help='time constant for Leaky Integrate and Fire (LIF) model: 20e-3 for dt=1, 50e-3 for dt=4, 100e-3 for dt=8')

parser.add_argument('--num_enc_layers', type=int, default=2, help='number of transformer encoder layers')
parser.add_argument('--num_dec_layers', type=int, default=2, help='number of transformer decoder layers')

parser.add_argument('--no_mixed_precision', dest='mixed_precision', action='store_false',
                    help='disable mixed precision (default is ON)')
parser.set_defaults(mixed_precision=True)

parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--dt', type=int, default=1, help='time interval (1, 4, or 8)')
parser.add_argument('--sp_threshold', type=float, default=0.75, choices=[0.75, 0.5],
                    help='spike threshold: 0.75 for dt=1, 0.5 for dt=4 or 8')

parser.add_argument('--num_bins', type=int, default=10, help='number of temporal bins for voxel grid')

parser.add_argument('--dsec_train_dir', default='/media/pzha9599/Software/dataset/dsec/train', help='Path to DSEC training data')
parser.add_argument('--dsec_test_dir', default='/media/windows_data/code/research/dataset/Event/dsec/test', help='Path to DSEC testing data')

parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup epochs for learning rate scheduler')
parser.add_argument('--eval_int', type=int, default=3, help='evaluation interval')
parser.add_argument('--max_fail_times', type=int, default=5, help='maximum failure times')

parser.add_argument('--save_thred', type=float, default=1.05, help='threashold for saving the checkpoint')

parser.add_argument('--train_host', default='local', choices=['local', 'h200'],
                    help='Host environment to determine dataset paths')
parser.add_argument('--data_ratio', type=float, default=1.0, 
                    help='Fraction of the dataset to use for rapid testing (e.g., 0.5 for half)')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vis_resolution = 256
sp_threshold = args.sp_threshold

div_flow = 20.0

if args.train_host == 'local':
    base_dir = '/media/windows_data/code/research'
else:
    base_dir = '/scratch/let-flownet'

save_dir = f'{base_dir}/outputs/let_flownet_voxel_multiscale_ilif_edc_loss_dt{args.dt}_output'

arch = "let_flownet_voxel"

epochs = 100
iter_g = 0

# ---> BACKWARD COMPATIBILITY: Dynamic Batching <---
# DSEC requires gradient accumulation to fit the 480x640 tensor in VRAM.
# MVSEC and UZH-FPV (256x256) remain at true batch size 8 for maximum GPU efficiency.
batch_size = 2
accumulate_steps = 4


def train(train_loader, model, optimizer, epoch, train_writer, scaler, accumulate_steps):
    global iter_g, args, vis_resolution, sp_threshold
    np.set_printoptions(precision=2)
    losses = AverageMeter()

    # switch to train mode
    model.train()

    multiscale_weights = [0.01, 0.02, 0.08, 1.0]
    print_freq = 100
    valid_batches = 0

    for i_batch, data in enumerate(train_loader, 0):
        voxel_tensor, gt_flow, gt_mask = data

        voxel_nonzero_count = torch.count_nonzero(voxel_tensor)
        if i_batch % 100 == 0:
            print(f"Batch {i_batch} received. Count of non-zero voxels: {voxel_nonzero_count}")

        # check if there are any non-zero elements
        if voxel_nonzero_count > 0:
            print_details = valid_batches % print_freq == 0

            event_data = voxel_tensor.to(device)

            with torch.amp.autocast('cuda', enabled=args.mixed_precision, dtype=torch.bfloat16):
                flow_predictions = model(event_data, sp_threshold)

            flow_preds_fp32 = [f.float() for f in flow_predictions]

            supervised_loss = supervised_loss_multiscale(
                flow_preds_fp32,
                (gt_flow.to(device).float() / div_flow),
                gt_mask.to(device).float(),
                weights=multiscale_weights
            )
            loss_metric = supervised_loss
            loss_name = 'supervised_loss'

            smoothness_loss = smooth_loss(flow_preds_fp32)
            total_loss = loss_metric + smoothness_loss
            loss = total_loss / accumulate_steps

            # --- MIXED PRECISION BACKWARD PASS ---
            # The scaler will compute the loss gradients in safe FP32, and automatically 
            # cast them back to FP16 when they flow backwards into the network layers.
            scaler.scale(loss).backward()

            # ---> GRADIENT ACCUMULATION: Only step every N batches <---
            if (i_batch + 1) % accumulate_steps == 0 or (i_batch + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # record loss and EPE (Multiply loss back up so TensorBoard graphs remain accurate)
            train_writer.add_scalar('train_loss/total_loss', loss.item() * accumulate_steps, iter_g)
            train_writer.add_scalar(f'train_loss/{loss_name}', loss_metric.item(), iter_g)
            train_writer.add_scalar('train_loss/smoothness_loss', smoothness_loss.item(), iter_g)
                
            losses.update(total_loss.item(), event_data.size(0))

            if print_details:
                now = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M-%S")
                print(f'Time: {now}, Epoch: [{epoch}][{batch_size * i_batch}/{batch_size * len(train_loader)}], Loss: {losses.val:.2f}, {loss_name}: {loss_metric.item():.2f}, smoothness_loss: {smoothness_loss.item():.2f}')
                print('-------------------------------------------------------')

            iter_g += 1
            valid_batches += 1

    return losses.avg


def validate(test_loader, model, epoch, output_writers, current_test_src_file, current_test_gt_file):
    global args, vis_resolution, sp_threshold
    d_label = h5py.File(current_test_gt_file, 'r')
    gt_temp = np.float32(d_label['davis']['left']['flow_dist'])
    gt_ts_temp = np.float64(d_label['davis']['left']['flow_dist_ts'])
    d_label = None

    d_set = h5py.File(current_test_src_file, 'r')
    gray_image = d_set['davis']['left']['image_raw']

    # switch to evaluate mode
    model.eval()

    epe_sum = 0.
    ae_sum = 0.
    pe1_sum = 0.
    pe2_sum = 0.
    pe3_sum = 0.
    total_points = 0.
    iters = 0.
    scale = 1

    print_freq = 100

    for i_batch, data in enumerate(test_loader, 0):
        voxel_tensor, ts_f, ts_l = data

        # check if there are any non-zero elements
        if torch.count_nonzero(voxel_tensor) > 0:
            event_data = voxel_tensor.to(device)

            # compute output
            output = model(event_data, sp_threshold)

            # ---> Extract final scale if using Multi-Scale <---
            if isinstance(output, list):
                output = output[-1]  # Extract flow3 (the final 256x256 prediction)

            output_temp = output.cpu()

            # ---> Infer dimensions directly from the loaded tensor <---
            input_h, input_w = event_data.size(2), event_data.size(3)

            # Interpolate natively in PyTorch to match the actual input
            output_resized = torch.nn.functional.interpolate(
                output_temp, size=(input_h, input_w), mode='bilinear', align_corners=False
            )

            # ---> CRITICAL FIX: Scale the flow magnitude by the spatial upsample factor <---
            scale_h = input_h / output_temp.size(2)
            scale_w = input_w / output_temp.size(3)

            # Only scale the translation parameters (v_x, v_y), NOT angular rotation (omega)
            output_resized[:, 0, :, :] *= scale_w
            output_resized[:, 1, :, :] *= scale_h
            
            # ---> Remove SE(2), Slice (u,v), and Apply Domain Scaling <---
            pred_flow = (output_resized[0, :2, :, :] * div_flow).permute(1, 2, 0).numpy()

            u_gt_all = gt_temp[:, 0, :, :].copy()
            v_gt_all = gt_temp[:, 1, :, :].copy()

            # DSEC Parser stores invalid pixels (0 in 16-bit PNG) as -256.0.
            # We must zero them out so the validity mask correctly ignores them.
            invalid_mask = (u_gt_all == -256.0) & (v_gt_all == -256.0)
            u_gt_all[invalid_mask] = 0.0
            v_gt_all[invalid_mask] = 0.0

            u_gt, v_gt = estimate_corresponding_gt_flow(
                u_gt_all, v_gt_all, gt_ts_temp, ts_f.numpy(), ts_l.numpy())
            gt_flow = np.stack((u_gt, v_gt), axis=2)

            # Mask derivation for Metric Calculation & Visualization
            mask_tensor = torch.sum((event_data[0] != 0).float(), dim=(0, 3)).cpu()
            mask_temp_np = mask_tensor.numpy() > 0

            #   ----------- Visualization
            if epoch < 0 and not torch.cuda.is_available():
                # Clean conversion to uint8
                spike_image = mask_temp_np.astype(np.uint8) * 255
                cv2.imshow('Spike Image', spike_image)

                gray = cv2.resize(
                    gray_image[i_batch], (scale*vis_resolution, scale * vis_resolution), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Gray Image', cv2.cvtColor(
                    gray, cv2.COLOR_BGR2RGB))

                # Extract to a clean NumPy array ONCE
                out_temp = output_temp.cpu().detach().numpy()
                
                # Directly slice the NumPy array (no np.array() wrappers)
                x_flow = cv2.resize(out_temp[0, 0, :, :], (
                    scale * vis_resolution, scale * vis_resolution), interpolation=cv2.INTER_LINEAR)
                y_flow = cv2.resize(out_temp[0, 1, :, :], (
                    scale * vis_resolution, scale * vis_resolution), interpolation=cv2.INTER_LINEAR)
                rgb_flow = flow_viz_np(x_flow, y_flow)
                cv2.imshow('Predicted Flow', cv2.cvtColor(
                    rgb_flow, cv2.COLOR_BGR2RGB))

                gt_flow_x = cv2.resize(
                    gt_flow[:, :, 0], (scale * vis_resolution, scale * vis_resolution), interpolation=cv2.INTER_LINEAR)
                gt_flow_y = cv2.resize(
                    gt_flow[:, :, 1], (scale * vis_resolution, scale * vis_resolution), interpolation=cv2.INTER_LINEAR)
                gt_flow_large = flow_viz_np(gt_flow_x, gt_flow_y)
                cv2.imshow('GT Flow', cv2.cvtColor(
                    gt_flow_large, cv2.COLOR_BGR2RGB))

                x_flow_masked = cv2.resize(out_temp[0, 0, :, :] * mask_temp_np, (
                    scale * vis_resolution, scale * vis_resolution), interpolation=cv2.INTER_LINEAR)
                y_flow_masked = cv2.resize(out_temp[0, 1, :, :] * mask_temp_np, (
                    scale * vis_resolution, scale * vis_resolution), interpolation=cv2.INTER_LINEAR)
                rgb_flow_masked = flow_viz_np(x_flow_masked, y_flow_masked)
                cv2.imshow('Masked Predicted Flow', cv2.cvtColor(
                    rgb_flow_masked, cv2.COLOR_BGR2RGB))

                # Calculate dynamic center offset for visualization crop to match network input exactly
                xoff_vis = max(0, (gt_flow.shape[1] - vis_resolution) // 2)
                yoff_vis = max(0, (gt_flow.shape[0] - vis_resolution) // 2)
                gt_flow_cropped = gt_flow[yoff_vis : yoff_vis + vis_resolution, xoff_vis : xoff_vis + vis_resolution, :]
                
                gt_flow_x_masked = gt_flow_cropped[:, :, 0] * mask_temp_np
                gt_flow_y_masked = gt_flow_cropped[:, :, 1] * mask_temp_np
                gt_flow_large_masked = flow_viz_np(
                    cv2.resize(gt_flow_x_masked, (scale * vis_resolution, scale * vis_resolution), interpolation=cv2.INTER_LINEAR),
                    cv2.resize(gt_flow_y_masked, (scale * vis_resolution, scale * vis_resolution), interpolation=cv2.INTER_LINEAR))
                cv2.imshow('Masked GT Flow', cv2.cvtColor(
                    gt_flow_large_masked, cv2.COLOR_BGR2RGB))

                cv2.waitKey(1)

            image_size = pred_flow.shape
            full_size = gt_flow.shape
            xcrop = image_size[1]
            ycrop = image_size[0]
            xsize = full_size[1]
            ysize = full_size[0]
            xoff = max(0, (xsize - xcrop) // 2)
            yoff = max(0, (ysize - ycrop) // 2)

            gt_flow = gt_flow[yoff : yoff + ycrop, xoff : xoff + xcrop, :]

            epe, ae, pe1, pe2, pe3, n_points = flow_error_dense(gt_flow, pred_flow, mask_temp_np, is_car=False)

            epe_sum += epe
            ae_sum += ae
            pe1_sum += pe1
            pe2_sum += pe2
            pe3_sum += pe3
            total_points += n_points

            if i_batch < len(output_writers):  # log first output of first batches
                output_writers[i_batch].add_image('Let FlowNet Outputs', flow2rgb(
                    div_flow * output_temp[0, :2], max_value=10), epoch)

            iters += 1

            now = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M-%S")

            # print evaluation progress
            if i_batch % print_freq == 0:
                print('-------------------------------------------------------')
                print(f'Time: {now}, i_batch: [{i_batch}/{len(test_loader)}]')
                print('Mean EPE: {:.3f}, Mean AE: {:.3f}°, 1PE: {:.2f}%, 2PE: {:.2f}%, 3PE: {:.2f}%, # pts: {:.2f}'
                    .format(epe_sum / iters, ae_sum / iters, pe1_sum / iters, pe2_sum / iters, pe3_sum / iters, n_points))

    seq_name = os.path.basename(current_test_src_file).replace('_data.hdf5', '')
    
    print(f'================ Validation Outcome: {seq_name} ===================')
    print(f'Time: {now}, epoch: {epoch}')
    print('Mean EPE: {:.3f}, Mean AE: {:.3f}°, 1PE: {:.2f}%, 2PE: {:.2f}%, 3PE: {:.2f}%, # Mean pts: {:.2f}'
        .format(epe_sum / iters, ae_sum / iters, pe1_sum / iters, pe2_sum / iters, pe3_sum / iters, total_points / iters))
    print('===============================================================')

    return epe_sum / iters


def main():
    global args
    # Initializations
    print(f"=> using device '{device}'")

    workers = 8
    best_EPE = -1
    val_fail_times = 0

    test_file_pairs = []
    test_envs = ['zurich_city_05_b', 'zurich_city_06_a', 'zurich_city_10_b', 'zurich_city_11_c']
    if args.data_ratio < 1.0:
        num_val = max(1, int(len(test_envs) * args.data_ratio))
        test_envs = test_envs[:num_val]

    for t_env in test_envs:
        test_file_pairs.append((
            os.path.join(args.dsec_test_dir, f"{t_env}_data.hdf5"),
            os.path.join(args.dsec_test_dir, f"{t_env}_gt.hdf5")
        ))

    test_loaders = []
    for t_src, t_gt in test_file_pairs:
        with h5py.File(t_gt, 'r') as d_label:
            gt_start = np.float64(d_label['davis']['left']['flow_dist_ts'])[0]
            
        t_dataset = DatasetTestDSEC(
            args.dt, t_src,
            gt_start_time=gt_start,
            num_bins=args.num_bins
        )
        t_loader = DataLoader(
            dataset=t_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            multiprocessing_context='spawn'
        )
        test_loaders.append((t_loader, t_src, t_gt))
        
    print(f"=> Created {len(test_loaders)} validation loader(s) for DSEC.")

    save_path = '{},bat{},lr{},bin{}'.format(
        arch,
        batch_size,
        args.lr,
        args.num_bins)

    timestamp = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M")
    save_path = os.path.join(timestamp, save_path)
    save_path = os.path.join(save_dir, save_path)
    os.makedirs(save_path, exist_ok=True)
    print(f'=> Everything will be saved to {save_path}')

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(
            os.path.join(save_path, 'test', str(i))))

    # create model
    if args.pretrained:
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        network_data = torch.load(args.pretrained, map_location)
        print(f"=> using pre-trained model '{arch}'")
    else:
        network_data = None
        print(f"=> creating model '{arch}'")

    model = let_flownet_voxel.__dict__[arch](args, device, network_data).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"=======================================================")
    print(f"=> Architecture: {args.num_enc_layers} Encoder / {args.num_dec_layers} Decoder Layers")
    print(f"=> Total Trainable Parameters: {total_params / 1e6:.2f} Million")
    print(f"=======================================================")

    model = torch.nn.DataParallel(model).to(device)

    cudnn.benchmark = True

    if args.evaluate:
        with torch.no_grad():
            total_EPE = 0
            for t_loader, t_src, t_gt in test_loaders:
                total_EPE += validate(t_loader, model, -1, output_writers, t_src, t_gt)
            mean_EPE = total_EPE / len(test_loaders)
            if len(test_loaders) > 1:
                print(f'================ Overall Validation Outcome ===================')
                print(f'Mean EPE across all {len(test_loaders)} sequences: {mean_EPE:.3f}')
                print('=============================================================')
        return

    assert (args.solver in ['adam', 'sgd'])
    print(f'=> setting {args.solver} solver')

    # 1. Safely extract PLIF decay parameters via .module
    # (Ensures the SNN temporal memory parameters are isolated)
    alpha_params = [
        model.module.alpha1, model.module.alpha2, model.module.alpha3, model.module.alpha4
    ]
    alpha_param_ids = list(map(id, alpha_params))

    # 2. Extract and filter bias/weight parameters to EXCLUDE the alpha params
    bias_params = [p for p in model.module.bias_parameters() if id(p) not in alpha_param_ids]
    weight_params = [p for p in model.module.weight_parameters() if id(p) not in alpha_param_ids]

    # ---> BACKWARD COMPATIBILITY: Dynamic SNN Learning Rate Scale <---
    # MVSEC and UZH-FPV keep the 0.01x bottleneck. DSEC gets full 1.0x velocity.
    snn_lr_scale = 1.0

    if args.solver == 'adam':
        optimizer = torch.optim.Adam([
            {'params': bias_params, 'weight_decay': 0.0},
            {'params': weight_params, 'weight_decay': 4e-4},
            {'params': alpha_params, 'lr': args.lr * snn_lr_scale, 'weight_decay': 0.0} 
        ], lr=args.lr)
        
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': bias_params, 'weight_decay': 0.0},
            {'params': weight_params, 'weight_decay': 4e-4},
            {'params': alpha_params, 'lr': args.lr * snn_lr_scale, 'weight_decay': 0.0}
        ], lr=args.lr, momentum=0.9)

    # Conditional Scheduler Setup
    if args.warmup_epochs > 0:
        # Warmup for first n epochs, then multistep decay
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs
        )
        scheduler_multistep = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[15, 30, 45, 60, 80], gamma=0.5
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler_warmup, scheduler_multistep], milestones=[args.warmup_epochs]
        )
    else:
        # TRUE 0-Warmup: Immediately start at base LR and only apply multistep decay
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[15, 30, 45, 60, 80], gamma=0.5
        )
        
    # ---> FIX: Physically fast-forward the scheduler to sync the optimizer's internal LRs <---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress harmless PyTorch 'step before optimizer' warning
        for _ in range(args.start_epoch):
            scheduler.step()

    # Use strict rigid transformations to preserve SNN spike density and physical scaling
    co_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5)
    ])

    import glob
    dsec_files = glob.glob(os.path.join(args.dsec_train_dir, "*_data.hdf5"))
    
    # SOTA Protocol: Exclude these from training to act as the local test split
    hold_outs = ['zurich_city_05_b', 'zurich_city_06_a', 'zurich_city_10_b', 'zurich_city_11_c']
    
    # Remove validation hold-outs from the available training pool
    dsec_files = [f for f in dsec_files if not any(val_seq in f for val_seq in hold_outs)]
    
    if args.data_ratio < 1.0:
        num_train = max(1, int(len(dsec_files) * args.data_ratio))
        dsec_files = dsec_files[:num_train]
        
    train_loader = []
    for dataset_path in dsec_files:
        gt_path = dataset_path.replace('_data.hdf5', '_gt.hdf5')
        if not os.path.exists(gt_path):
            print(f"Skipping {dataset_path}: No GT file found for supervised training")
            continue
        print(f"Loading DSEC Supervised Train dataset {dataset_path}...")
        single_dataset = DatasetTrainDSEC_Supervised(
            args.dt,
            dataset_path,
            gt_path,
            transform=co_transform,
            num_bins=args.num_bins
        )
        train_loader.append(
            DataLoader(
                dataset=single_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                pin_memory=True,
                drop_last=True,
                multiprocessing_context='spawn'
            )
        )

    # Initialize Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision)

    for epoch in range(args.start_epoch, epochs):

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")

        # Shuffle the order we read the HDF5 sequence files every epoch
        random.shuffle(train_loader)
        
        epoch_loss = 0
        for loader in train_loader:
            # Train fully on one file before moving to the next
            loss = train(loader, model, optimizer, epoch, train_writer, scaler, accumulate_steps)
            epoch_loss += loss
            
        train_loss = epoch_loss / len(train_loader)

        train_writer.add_scalar('mean_train_loss', train_loss, epoch)

        print(f"Mean Training Loss: {train_loss:.3f} of epoch {epoch}")
        print('-------------------------------------------------------')

        scheduler.step()

        # Test at every n epoch during training
        if (epoch + 1) % args.eval_int == 0:
            # evaluate on validation set
            with torch.no_grad():
                total_EPE = 0
                for t_loader, t_src, t_gt in test_loaders:
                    total_EPE += validate(t_loader, model, epoch, output_writers, t_src, t_gt)
                EPE = total_EPE / len(test_loaders)
            if len(test_loaders) > 1:
                print(f'================ Overall Validation Outcome (Epoch {epoch}) ===================')
                print(f'Mean EPE across all {len(test_loaders)} sequences: {EPE:.3f}')
                print('=============================================================================')

            test_writer.add_scalar('mean_val_EPE', EPE, epoch)

            if best_EPE < 0:
                best_EPE = EPE

            is_best = EPE < best_EPE
            best_EPE = min(EPE, best_EPE)

            if EPE < args.save_thred:
                filename = f'checkpoint_epoch_{epoch + 1}_{EPE}.pth.tar'
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': arch,
                    'state_dict': model.module.state_dict(),
                    'best_EPE': best_EPE,
                    'div_flow': div_flow,
                    'num_bins': args.num_bins
                }, is_best, save_path, filename=filename)

            # check if exit criteria is met
            if is_best:
                val_fail_times = 0
            else:
                val_fail_times += 1

            if val_fail_times >= args.max_fail_times:
                print(
                    "Epoch {}: validation failed for consective {} times".format(
                        epoch, val_fail_times
                    )
                )
                break


if __name__ == '__main__':
    main()
