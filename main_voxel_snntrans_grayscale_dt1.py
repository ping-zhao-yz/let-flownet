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
from torch.utils.data import ConcatDataset, DataLoader

from util.loss_util import AverageMeter
from util.flow_util import flow2rgb, flow_viz_np, save_checkpoint

from datasets.voxel.dataset_dtx import DatasetTest, DatasetTrain
from models import let_flownet_voxel
from loss.multiscaleloss import estimate_corresponding_gt_flow, flow_error_dense, smooth_loss
from loss.photometric_loss_backward import photometric_loss_multiscale

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

parser.add_argument('--tau', default=20*1e-3, help='time constant for Leaky Integrate and Fire (LIF) model')

parser.add_argument('--num_enc_layers', default=2, help='number of transformer encoder layers')
parser.add_argument('--num_dec_layers', default=2, help='number of transformer decoder layers')

parser.add_argument('--no_mixed_precision', dest='mixed_precision', action='store_false',
                    help='disable mixed precision (default is ON)')
parser.set_defaults(mixed_precision=True)

parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--dt', type=int, default=1, help='time interval (1, 4, or 8)')
parser.add_argument('--sp_threshold', type=float, default=0.75, help='spike threshold')

parser.add_argument('--num_bins', type=int, default=10, help='number of temporal bins for voxel grid')

parser.add_argument('--train_dataset', default='mvsec', choices=['mvsec', 'uzh-fpv'],
                    help='dataset for training')

parser.add_argument('--train_env', default='outdoor_day2', help='train env (outdoor_day1 or outdoor_day2)')
parser.add_argument('--test_env', default='indoor_flying1', help='test env (indoor_flying1, indoor_flying2, or indoor_flying3)')

parser.add_argument('--lr', type=float, default=1e-5, choices=[1e-4, 1e-5, 1e-6],
                    help='learning rate: 1e-4 for training from scratch, 1e-5/1e-6 for fine-tuning, both with 3 epochs warmup')
parser.add_argument('--eval_int', type=int, default=3, choices=[3, 1, 1],
                    help='evaluation interval: 3 for training from scratch; 1 for domain bridge; 1 for fine tuning')
parser.add_argument('--max_fail_times', type=int, default=5, choices=[5, 4, 10, 15, 30],
                    help='maximum failure times: 5 for training from scratch; 4 for domain bridge; 10 for fine tuning')
parser.add_argument('--warmup_epochs', type=int, default=3, choices=[3, 3, 0],
                    help='warmup epochs for learning rate scheduler: 3 for training from scratch, 3 for domain bridge; 0 for fine tuning')

parser.add_argument('--save_thred', type=float, default=1.05,
                    help='threashold for saving the checkpoint')

args = parser.parse_args()

# Initializations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=> using device '{device}'")

image_resize = 256
sp_threshold = args.sp_threshold
div_flow = 1

src_file_dir = '/media/windows_data/code/research/dataset/Event/mvsec/original'

train_env = args.train_env
test_env = args.test_env

train_src_file = src_file_dir + '/' + train_env + '/' + train_env + "_data.hdf5"
test_src_file = src_file_dir + '/' + test_env + '/' + test_env + "_data.hdf5"
test_gt_file = src_file_dir + '/' + test_env + '/' + test_env + "_gt.hdf5"

uzh_fpv_dataset_path = '/media/windows_data/code/research/dataset/Event/uzh-fpv/data/'

save_dir = 'let_flownet_voxel_multiscale_dt1_output'

arch = "let_flownet_voxel"

epochs = 100
batch_size = 8
iter_g = 0


def train(train_loader, model, optimizer, epoch, train_writer, scaler):
    global iter_g, args, image_resize, sp_threshold
    np.set_printoptions(precision=2)
    losses = AverageMeter()

    # switch to train mode
    model.train()

    multiscale_weights = [0.01, 0.02, 0.08, 1.0]
    print_freq = 100
    valid_batches = 0

    for i_batch, data in enumerate(train_loader, 0):
        voxel_tensor, former_gray, latter_gray = data

        voxel_nonzero_count = torch.count_nonzero(voxel_tensor)
        if i_batch % 100 == 0:
            print(f"Batch {i_batch} received. Count of non-zero voxels: {voxel_nonzero_count}")

        # check if there are any non-zero elements
        if voxel_nonzero_count > 0:
            print_details = valid_batches % print_freq == 0

            # No need for initInputRepresentation; shape is already [Batch, 2, H, W, num_bins]
            event_data = voxel_tensor.to(device)

            # --- MIXED PRECISION FORWARD PASS ---
            with torch.amp.autocast('cuda', enabled=args.mixed_precision, dtype=torch.bfloat16):
                # 1. Compute output (SNN + Transformer run in ultra-fast FP16)
                flow_predictions = model(event_data, image_resize, sp_threshold)

            # --- FORCE LOSS CALCULATION TO FP32 ---
            # 2. Step OUTSIDE the autocast block and explicitly cast to float32.
            # This prevents grid_sample and division underflow NaNs in the loss!
            flow_preds_fp32 = [f.float() for f in flow_predictions]
            
            event_mask = (torch.sum((event_data != 0).float(), dim=(1, 4)) > 0).float()
            
            photometric_loss = photometric_loss_multiscale(
                former_gray[:, 0, :, :].to(device).float(), 
                latter_gray[:, 0, :, :].to(device).float(), 
                event_mask, 
                flow_preds_fp32, 
                device, 
                print_details, 
                weights=multiscale_weights
            )

            # Smoothness loss
            smoothness_loss = smooth_loss(flow_preds_fp32)

            # total_loss
            loss = photometric_loss + smoothness_loss

            optimizer.zero_grad()
            
            # --- MIXED PRECISION BACKWARD PASS ---
            # The scaler will compute the loss gradients in safe FP32, and automatically 
            # cast them back to FP16 when they flow backwards into the network layers.
            scaler.scale(loss).backward()

            # Unscale the gradients BEFORE clipping to ensure the max_norm threshold is accurate
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()

            # record loss and EPE
            train_writer.add_scalar('train_loss', loss.item(), iter_g)
            losses.update(loss.item(), event_data.size(0))

            if print_details:
                now = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M-%S")
                print(f'Time: {now}, Epoch: [{epoch}][{batch_size * i_batch}/{batch_size * len(train_loader)}], Loss: {losses}, photometric_loss: {round(photometric_loss.item(), 2)}, smoothness_loss: {smoothness_loss.item():.2f}')
                print('-------------------------------------------------------')

            iter_g += 1
            valid_batches += 1

    return losses.avg


def validate(test_loader, model, epoch, output_writers):
    global args, image_resize, sp_threshold
    d_label = h5py.File(test_gt_file, 'r')
    gt_temp = np.float32(d_label['davis']['left']['flow_dist'])
    gt_ts_temp = np.float64(d_label['davis']['left']['flow_dist_ts'])
    d_label = None

    d_set = h5py.File(test_src_file, 'r')
    gray_image = d_set['davis']['left']['image_raw']

    # switch to evaluate mode
    model.eval()

    AEE_sum = 0.
    AEE_sum_sum = 0.
    AEE_sum_gt = 0.
    AEE_sum_sum_gt = 0.
    total_points = 0.
    percent_Outlier_sum = 0.
    iters = 0.
    scale = 1

    print_freq = 100

    for i_batch, data in enumerate(test_loader, 0):
        voxel_tensor, ts_f, ts_l = data

        # Only for outdoor_day1, limit to 800 frames to match Spike-FlowNet
        if 'outdoor_day1' in test_env and i_batch >= 800:
            break

        # check if there are any non-zero elements
        if torch.count_nonzero(voxel_tensor) > 0:
            event_data = voxel_tensor.to(device)

            # compute output
            output = model(event_data, image_resize, sp_threshold)

            # ---> Extract final scale if using Multi-Scale <---
            if isinstance(output, list):
                output = output[-1]  # Extract flow3 (the final 256x256 prediction)

            output_temp = output.cpu()

            # Interpolate natively in PyTorch
            output_resized = torch.nn.functional.interpolate(
                output_temp, size=(image_resize, image_resize), mode='bilinear', align_corners=False
            )

            # ---> CRITICAL FIX: Scale the flow magnitude by the spatial upsample factor <---
            scale_h = image_resize / output_temp.size(2)
            scale_w = image_resize / output_temp.size(3)
            output_resized[:, 0, :, :] *= scale_w
            output_resized[:, 1, :, :] *= scale_h
            
            # Permute from [Channels, H, W] to [H, W, Channels] and convert to clean numpy
            pred_flow = output_resized[0].permute(1, 2, 0).numpy()

            u_gt_all = gt_temp[:, 0, :, :]
            v_gt_all = gt_temp[:, 1, :, :]

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
                    gray_image[i_batch], (scale*image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Gray Image', cv2.cvtColor(
                    gray, cv2.COLOR_BGR2RGB))

                # Extract to a clean NumPy array ONCE
                out_temp = output_temp.cpu().detach().numpy()
                
                # Directly slice the NumPy array (no np.array() wrappers)
                x_flow = cv2.resize(out_temp[0, 0, :, :], (
                    scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                y_flow = cv2.resize(out_temp[0, 1, :, :], (
                    scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                rgb_flow = flow_viz_np(x_flow, y_flow)
                cv2.imshow('Predicted Flow', cv2.cvtColor(
                    rgb_flow, cv2.COLOR_BGR2RGB))

                gt_flow_x = cv2.resize(
                    gt_flow[:, :, 0], (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                gt_flow_y = cv2.resize(
                    gt_flow[:, :, 1], (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                gt_flow_large = flow_viz_np(gt_flow_x, gt_flow_y)
                cv2.imshow('GT Flow', cv2.cvtColor(
                    gt_flow_large, cv2.COLOR_BGR2RGB))

                x_flow_masked = cv2.resize(out_temp[0, 0, :, :] * mask_temp_np, (
                    scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                y_flow_masked = cv2.resize(out_temp[0, 1, :, :] * mask_temp_np, (
                    scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                rgb_flow_masked = flow_viz_np(x_flow_masked, y_flow_masked)
                cv2.imshow('Masked Predicted Flow', cv2.cvtColor(
                    rgb_flow_masked, cv2.COLOR_BGR2RGB))

                gt_flow_cropped = gt_flow[2: -2, 45: -45]
                gt_flow_x_masked = cv2.resize(
                    gt_flow_cropped[:, :, 0] * mask_temp_np, (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                gt_flow_y_masked = cv2.resize(
                    gt_flow_cropped[:, :, 1] * mask_temp_np, (scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                gt_flow_large_masked = flow_viz_np(
                    gt_flow_x_masked, gt_flow_y_masked)
                cv2.imshow('Masked GT Flow', cv2.cvtColor(
                    gt_flow_large_masked, cv2.COLOR_BGR2RGB))

                cv2.waitKey(1)

            image_size = pred_flow.shape
            full_size = gt_flow.shape
            xcrop = image_size[1]
            ycrop = image_size[0]
            xsize = full_size[1]
            ysize = full_size[0]
            xoff = (xsize - xcrop) // 2
            yoff = (ysize - ycrop) // 2

            gt_flow = gt_flow[yoff: -yoff, xoff: -xoff, :]

            is_car_flag = 'outdoor' in test_env

            AEE, percent_Outlier, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = flow_error_dense(
                gt_flow, pred_flow, mask_tensor, is_car=is_car_flag)

            AEE_sum = AEE_sum + div_flow * AEE
            AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

            AEE_sum_gt = AEE_sum_gt + div_flow * AEE_gt
            AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

            percent_Outlier_sum += percent_Outlier
            total_points += n_points

            if i_batch < len(output_writers):  # log first output of first batches
                output_writers[i_batch].add_image('SpikeT FlowNet Outputs', flow2rgb(
                    div_flow * output_temp[0], max_value=10), epoch)

            iters += 1

            now = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M-%S")

            # print evaluation progress
            if i_batch % print_freq == 0:
                print('-------------------------------------------------------')
                print(f'Time: {now}, i_batch: [{i_batch}/{len(test_loader)}]')
                print('Mean AEE: {:.3f}, sum AEE: {:.2f}, Mean AEE_gt: {:.2f}, sum AEE_gt: {:.2f}, Mean %Outlier: {:.3f}, # pts: {:.2f}'
                    .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters, percent_Outlier_sum / iters, n_points))

    print('================ Overall Validation Outcome ===================')
    print(f'Time: {now}, epoch: {epoch}')
    print('Mean AEE: {:.3f}, sum AEE: {:.2f}, Mean AEE_gt: {:.2f}, sum AEE_gt: {:.2f}, Mean %Outlier: {:.3f}, # Mean pts: {:.2f}'
        .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters, percent_Outlier_sum / iters, total_points / iters))
    print('===============================================================')

    return AEE_sum / iters


def main():
    global args

    workers = 8
    best_EPE = -1
    val_fail_times = 0

    d_label = h5py.File(test_gt_file, 'r')
    gt_start = np.float64(d_label['davis']['left']['flow_dist_ts'])[0]
    d_label.close()

    save_path = '{},{},epochs{},bat{},lr{}'.format(
        arch,
        args.solver,
        epochs,
        batch_size,
        args.lr)

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

    test_dataset = DatasetTest(args.dt, test_src_file, gt_start_time=gt_start, num_bins=args.num_bins)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=workers)

    # create model
    if args.pretrained:
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        network_data = torch.load(args.pretrained, map_location)
        print(f"=> using pre-trained model '{arch}'")
    else:
        network_data = None
        print(f"=> creating model '{arch}'")

    model = let_flownet_voxel.__dict__[arch](args, device, network_data).to(device)
    model = torch.nn.DataParallel(model).to(device)

    cudnn.benchmark = True

    if args.evaluate:
        with torch.no_grad():
            best_EPE = validate(test_loader, model, -1, output_writers)
        return

    assert (args.solver in ['adam', 'sgd'])
    print(f'=> setting {args.solver} solver')

    # 1. Safely extract PLIF decay parameters via .module
    alpha_params = [
        model.module.alpha1, model.module.alpha2, model.module.alpha3, model.module.alpha4
    ]
    alpha_param_ids = list(map(id, alpha_params))

    # 2. Extract and filter bias/weight parameters to EXCLUDE the alpha params
    bias_params = [p for p in model.module.bias_parameters() if id(p) not in alpha_param_ids]
    weight_params = [p for p in model.module.weight_parameters() if id(p) not in alpha_param_ids]

    if args.solver == 'adam':
        optimizer = torch.optim.Adam([
            {'params': bias_params, 'weight_decay': 0.0},
            {'params': weight_params, 'weight_decay': 4e-4},
            {'params': alpha_params, 'lr': args.lr * 0.01, 'weight_decay': 0.0} # Ensure SNN decay parameters aren't flattened by L2
        ], lr=args.lr)
        
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': bias_params, 'weight_decay': 0.0},
            {'params': weight_params, 'weight_decay': 4e-4},
            {'params': alpha_params, 'lr': args.lr * 0.01, 'weight_decay': 0.0}
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

        # ---> Physically fast-forward the scheduler to sync the optimizer's internal LRs <---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress harmless PyTorch 'step before optimizer' warning
            for _ in range(args.start_epoch):
                scheduler.step()

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[15, 30, 45, 60, 80], gamma=0.5, last_epoch=args.start_epoch - 1
        )

    # Use strict rigid transformations to preserve SNN spike density and physical scaling
    co_transform = transforms.Compose([
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5)
    ])

    assert (args.train_dataset in ['mvsec', 'uzh-fpv'])

    if args.train_dataset == 'mvsec':
        train_datasets = DatasetTrain(
            args.dt,
            train_src_file,
            transform=co_transform,
            is_fine_tune=args.train_env=='outdoor_day2',
            num_bins=args.num_bins
        )
        train_loader = DataLoader(
            dataset=train_datasets,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers
        )
    elif args.train_dataset == 'uzh-fpv':
        uzh_datasets = [
            uzh_fpv_dataset_path + 'indoor_forward_3.h5',
            uzh_fpv_dataset_path + 'indoor_forward_5.h5',
            uzh_fpv_dataset_path + 'indoor_forward_6.h5',
            uzh_fpv_dataset_path + 'indoor_forward_7.h5',
            uzh_fpv_dataset_path + 'indoor_forward_8.h5',
            uzh_fpv_dataset_path + 'indoor_forward_9.h5',
            uzh_fpv_dataset_path + 'indoor_forward_10.h5',
            uzh_fpv_dataset_path + 'indoor_forward_11.h5',
            uzh_fpv_dataset_path + 'indoor_forward_12.h5'
        ]
        
        # Initialize individual DatasetTrain objects and store them in a list
        train_loader = []
        for dataset_path in uzh_datasets:
            print(f"Loading UZH FPV dataset {dataset_path}...")
            single_dataset = DatasetTrain(
                args.dt, 
                dataset_path, 
                transform=co_transform, 
                num_bins=args.num_bins
            )
            # Create a separate loader for EACH file
            single_loader = DataLoader(
                dataset=single_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=workers, 
                pin_memory=True, 
                drop_last=True
            )
            train_loader.append(single_loader)

    # Initialize Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision)

    for epoch in range(args.start_epoch, epochs):

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")

        if args.train_dataset == 'uzh-fpv':
            # Shuffle the order we read the 9 HDF5 files every epoch
            random.shuffle(train_loader)
            
            epoch_loss = 0
            for loader in train_loader:
                # Train fully on one file before moving to the next
                loss = train(loader, model, optimizer, epoch, train_writer, scaler)
                epoch_loss += loss
            
            train_loss = epoch_loss / len(train_loader)
        else:
            # Standard MVSEC single-loader logic
            train_loss = train(train_loader, model, optimizer, epoch, train_writer, scaler)

        train_writer.add_scalar('mean_train_loss', train_loss, epoch)

        print(f"Mean Training Loss: {train_loss:.3f} of epoch {epoch}")
        print('-------------------------------------------------------')

        scheduler.step()

        # Test at every n epoch during training
        if (epoch + 1) % args.eval_int == 0:
            # evaluate on validation set
            with torch.no_grad():
                EPE = validate(test_loader, model, epoch, output_writers)
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
                    'div_flow': div_flow
                }, is_best, save_path, filename=filename)

            # check if exit criteria is met
            if EPE < best_EPE:
                val_fail_times = 0
            else:
                val_fail_times += 1

            if val_fail_times >= args.max_fail_times:
                if args.train_dataset == 'mvsec':
                    print(
                        "Epoch {}: validation failed for consective {} times".format(
                            epoch, val_fail_times
                        )
                    )
                    break
                elif args.train_dataset == 'uzh-fpv':
                    print(
                        "Epoch {}: validation failed for consective {} times, still continue as this is pre-training on UZH-FPV dataset".format(
                            epoch, val_fail_times
                        )
                    )


if __name__ == '__main__':
    main()
