from loss import loss_photometric
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
import torch.distributed as dist

from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from util.loss_util import AverageMeter
from util.flow_util import flow2rgb, flow_viz_np, save_checkpoint

from datasets.voxel.dataset_unsupervised import DatasetTest, DatasetTrain
from models import let_flownet_voxel
from loss.metrics import estimate_corresponding_gt_flow, flow_error_dense
from loss.loss_photometric import smooth_loss, photometric_loss_multiscale
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

parser.add_argument('--iters', type=int, default=8, help='number of refinement iterations (set to 0 for single-pass models)')

parser.add_argument('--no_mixed_precision', dest='mixed_precision', action='store_false',
                    help='disable mixed precision (default is ON)')
parser.set_defaults(mixed_precision=True)

parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--dt', type=int, default=1, help='time interval (1, 4, or 8)')
parser.add_argument('--sp_threshold', type=float, default=0.75, choices=[0.75, 0.5],
                    help='spike threshold: 0.75 for dt=1, 0.5 for dt=4 or 8')

parser.add_argument('--num_bins', type=int, default=10, help='number of temporal bins for voxel grid')

parser.add_argument('--train_dataset', default='mvsec', choices=['mvsec', 'uzh-fpv'],
                    help='dataset for training')

parser.add_argument('--train_env', default='outdoor_day1', help='train env (outdoor_day1 or outdoor_day2)')
parser.add_argument('--test_env', default='indoor_flying1', help='test env (indoor_flying1, indoor_flying2, or indoor_flying3)')

parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup epochs for learning rate scheduler')
parser.add_argument('--eval_int', type=int, default=3, help='evaluation interval')
parser.add_argument('--max_fail_times', type=int, default=5, help='maximum failure times')

parser.add_argument('--save_thred', type=float, default=10.0,
                    help='threshold for saving intermediate epoch checkpoints (default: 10.0; model_best is always saved)')

parser.add_argument('--freeze_backbone_epochs', type=int, default=0,
                    help='Number of initial epochs to freeze coarse backbone and auto-detach coarse flow')
parser.add_argument('--backbone_lr_scale', type=float, default=0.5,
                    help='Learning rate multiplier for coarse SNN/TPA backbone relative to refiner head (default: 0.5, i.e., 1:2 ratio)')
parser.add_argument('--seq_loss_weight', type=float, default=1.0,
                    help='Weight for the full-resolution sequence refinement loss (default: 1.0)')

parser.add_argument('--train_host', default='local', choices=['local', 'h200'],
                    help='Host environment to determine dataset paths')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vis_resolution = 256
sp_threshold = args.sp_threshold

div_flow = 1.0

if args.train_host == 'local':
    base_dir = '/media/windows_data/code/research'
else:
    base_dir = '/scratch/let-flownet'

uzh_fpv_dataset_path = f'{base_dir}/dataset/Event/uzh-fpv/data/'

train_env = args.train_env
test_env = args.test_env

# UZH-FPV and MVSEC share the same validation dataset; UZH-FPV doesn't use training dataset configuration here
src_file_dir = f'{base_dir}/dataset/Event/mvsec/original'
train_src_file = src_file_dir + '/' + train_env + '/' + train_env + "_data.hdf5"
test_envs = [env.strip() for env in test_env.split(',')]
test_file_pairs = []
for t_env in test_envs:
    t_src_file = f'{src_file_dir}/{t_env}/{t_env}_data.hdf5'
    t_gt_file = f'{src_file_dir}/{t_env}/{t_env}_gt.hdf5'
    test_file_pairs.append((t_src_file, t_gt_file, t_env))

save_dir = f'{base_dir}/outputs/let_flownet_voxel_multipass_trans_dt{args.dt}_output'

arch = "let_flownet_voxel"

epochs = 100
iter_g = 0
batch_size = 8


def train(train_loader, model, optimizer, epoch, train_writer, scaler):
    global iter_g, args, vis_resolution, sp_threshold
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

            event_data = voxel_tensor.to(device)

            # Automatically detach coarse flow during frozen curriculum epochs
            should_detach = (epoch < args.freeze_backbone_epochs)

            # --- MIXED PRECISION FORWARD PASS ---
            with torch.amp.autocast('cuda', enabled=args.mixed_precision, dtype=torch.bfloat16):
                output = model(event_data, sp_threshold, iters=args.iters, detach_coarse=should_detach)
                if args.iters == 0:
                    flows_p1 = output
                    flow_sequence = []
                else:
                    flows_p1, flow_sequence = output

            # --- FORCE LOSS CALCULATION TO FP32 ---
            flows_p1_fp32 = [f.float() for f in flows_p1]
            seq_fp32 = [f.float() for f in flow_sequence]

            former_gray_tensor = former_gray[:, 0, :, :].to(device).float()
            latter_gray_tensor = latter_gray[:, 0, :, :].to(device).float()
            event_mask = (torch.sum((event_data != 0).float(), dim=(1, 4)) > 0).float()

            # 1. Base Multi-Scale Loss for the Coarse Global Pass
            loss_coarse = photometric_loss_multiscale(
                former_gray_tensor,
                latter_gray_tensor,
                event_mask, 
                flows_p1_fp32,
                device,
                print_details,
                weights=multiscale_weights
            )

            # 2. RAFT Sequence Loss for the Refinement Passes
            loss_seq = 0.0
            gamma = 0.8
            n_predictions = len(seq_fp32)
            
            # Compute RAFT gamma weights and normalize so they sum to 1.0
            weights_raw = [gamma ** (n_predictions - i - 1) for i in range(n_predictions)]
            weight_sum = sum(weights_raw)
            
            for i, f_pred in enumerate(seq_fp32):
                # Normalized weight
                i_weight = weights_raw[i] / weight_sum
                
                # Trick the multiscale function into evaluating a single scale at native 256x256
                i_loss = photometric_loss_multiscale(
                    former_gray_tensor,
                    latter_gray_tensor,
                    event_mask,
                    [f_pred],
                    device,
                    print_details=False,
                    weights=[args.seq_loss_weight]
                )
                loss_seq += i_weight * i_loss

            loss_metric = loss_coarse + loss_seq
            loss_metric_name = 'photometric_loss'

            # --- Smoothness Loss Scaling ---
            smooth_weight = 1.0

            if len(seq_fp32) > 0:
                refined_flow_full = seq_fp32[-1]
                raw_smooth = smooth_loss(flows_p1_fp32) + smooth_loss([refined_flow_full])
            else:
                raw_smooth = smooth_loss(flows_p1_fp32)
            
            smoothness_loss = raw_smooth * smooth_weight
            loss = loss_metric + smoothness_loss

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
            train_writer.add_scalar('train_loss/total_loss', loss.item(), iter_g)
            train_writer.add_scalar(f'train_loss/{loss_metric_name}', loss_metric.item(), iter_g)
            train_writer.add_scalar('train_loss/smoothness_loss', smoothness_loss.item(), iter_g)
                
            losses.update(loss.item(), event_data.size(0))

            if print_details:
                now = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M-%S")
                print(f'Time: {now}, Epoch: [{epoch}][{i_batch}/{len(train_loader)}], Loss: {losses.val:.2f}, {loss_metric_name}: {loss_metric.item():.2f}, smoothness_loss: {smoothness_loss.item():.2f}')
                print('-------------------------------------------------------')

            iter_g += 1
            valid_batches += 1

    return losses.avg


def validate(test_loader, model, epoch, output_writers, current_test_src_file, current_test_gt_file, current_test_env):
    global args, vis_resolution, sp_threshold
    d_label = h5py.File(current_test_gt_file, 'r')
    gt_temp = np.float32(d_label['davis']['left']['flow_dist'])
    gt_ts_temp = np.float64(d_label['davis']['left']['flow_dist_ts'])
    d_label = None

    d_set = h5py.File(current_test_src_file, 'r')
    gray_image = d_set['davis']['left']['image_raw']

    # switch to evaluate mode (use underlying module to prevent DataParallel batch_size=1 scatter crash)
    eval_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    eval_model.eval()

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

        # Only for outdoor_day1, limit to 800 frames to match Spike-FlowNet
        if 'outdoor_day1' in current_test_env and i_batch >= 800:
            break

        # check if there are any non-zero elements
        if torch.count_nonzero(voxel_tensor) > 0:
            event_data = voxel_tensor.to(device)

            # compute output directly on primary device
            output = eval_model(event_data, sp_threshold, iters=args.iters)

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

            is_car_flag = 'outdoor' in current_test_env

            epe, ae, pe1, pe2, pe3, n_points = flow_error_dense(gt_flow, pred_flow, mask_temp_np, is_car=is_car_flag)

            # =========================================================================
            # Qualitative Frame Export (for Paper / Thesis Figures)
            # =========================================================================
            # Filter criteria: High ground-truth motion AND low prediction error
            num_valid_events = np.count_nonzero(mask_temp_np)
            if num_valid_events > 200:
                gt_magnitude = np.sqrt(gt_flow[:, :, 0]**2 + gt_flow[:, :, 1]**2)
                mean_gt_motion = np.mean(gt_magnitude[mask_temp_np])

                # Current frame EPE is strictly below 0.7 with significant motion (> 3.0 px)
                if epe < 0.70 and mean_gt_motion > 3.0:
                    dump_dir = "./qualitative_hunt"
                    os.makedirs(dump_dir, exist_ok=True)
                    seq_name = os.path.basename(current_test_src_file).replace('_data.hdf5', '')
                    frame_id = int(iters)

                    # 1. Reference Grayscale Image (APS)
                    gray = cv2.resize(gray_image[i_batch], (vis_resolution, vis_resolution), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(f"{dump_dir}/{seq_name}_f{frame_id:04d}_1_APS.png", gray)

                    # 2. Event Surface / Spike Mask
                    spike_vis = (mask_temp_np.astype(np.uint8) * 255)
                    spike_vis = cv2.resize(spike_vis, (vis_resolution, vis_resolution), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(f"{dump_dir}/{seq_name}_f{frame_id:04d}_2_Events.png", spike_vis)

                    # 3. Ground Truth Flow (Masked & Full)
                    gt_x = cv2.resize(gt_flow[:, :, 0], (vis_resolution, vis_resolution), interpolation=cv2.INTER_LINEAR)
                    gt_y = cv2.resize(gt_flow[:, :, 1], (vis_resolution, vis_resolution), interpolation=cv2.INTER_LINEAR)
                    gt_flow_bgr = flow_viz_np(gt_x, gt_y)
                    cv2.imwrite(f"{dump_dir}/{seq_name}_f{frame_id:04d}_3_GT.png", gt_flow_bgr)

                    # Masked GT (standard format for event-vision papers)
                    mask_resized = cv2.resize(mask_temp_np.astype(np.float32), (vis_resolution, vis_resolution), interpolation=cv2.INTER_NEAREST) > 0.5
                    gt_masked_bgr = gt_flow_bgr.copy()
                    gt_masked_bgr[~mask_resized] = 255  # Clean white background outside events
                    cv2.imwrite(f"{dump_dir}/{seq_name}_f{frame_id:04d}_3_GT_masked.png", gt_masked_bgr)

                    # 4. Our Multi-Pass Flow Prediction (Aligned & Scaled)
                    pred_x = cv2.resize(pred_flow[:, :, 0], (vis_resolution, vis_resolution), interpolation=cv2.INTER_LINEAR)
                    pred_y = cv2.resize(pred_flow[:, :, 1], (vis_resolution, vis_resolution), interpolation=cv2.INTER_LINEAR)
                    pred_flow_bgr = flow_viz_np(pred_x, pred_y)
                    cv2.imwrite(f"{dump_dir}/{seq_name}_f{frame_id:04d}_4_Ours_EPE{epe:.2f}.png", pred_flow_bgr)
            # =========================================================================

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
    global args, device, is_distributed, is_main_process, local_rank
    
    # ------------------ DYNAMIC DISTRIBUTED / SINGLE-GPU SETUP ------------------
    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    
    if is_distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = (local_rank == 0)
    if is_main_process:
        print(f"=> Distributed: {is_distributed} (World Size: {world_size}), Device: {device}")
    # -----------------------------------------------------------------------------

    workers = 8
    best_EPE = -1
    val_fail_times = 0

    test_loaders = []
    for t_src, t_gt, t_env in test_file_pairs:
        with h5py.File(t_gt, 'r') as d_label:
            gt_start = np.float64(d_label['davis']['left']['flow_dist_ts'])[0]
            
        t_dataset = DatasetTest(
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
        test_loaders.append((t_loader, t_src, t_gt, t_env))
        
    print(f"=> Created {len(test_loaders)} validation loader(s).")

    save_path = '{},bat{},lr{},bin{},itr{}'.format(
        arch,
        batch_size,
        args.lr,
        args.num_bins,
        args.iters)

    timestamp = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M-%S")
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
        network_data = torch.load(args.pretrained, map_location, weights_only=False)
        print(f"=> using pre-trained model '{arch}'")
    else:
        network_data = None
        print(f"=> creating model '{arch}'")

    model = let_flownet_voxel.__dict__[arch](args, device, network_data).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
        print(f"=======================================================")
        print(f"=> Architecture: {args.num_enc_layers} Encoder / {args.num_dec_layers} Decoder Layers")
        print(f"=> Total Trainable Parameters: {total_params / 1e6:.2f} Million")
        print(f"=======================================================")

    model = model.to(device)
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    cudnn.benchmark = True

    if args.evaluate:
        with torch.no_grad():
            total_EPE = 0
            for t_loader, t_src, t_gt, t_env in test_loaders:
                total_EPE += validate(t_loader, model, -1, output_writers, t_src, t_gt, t_env)
            mean_EPE = total_EPE / len(test_loaders)
            if len(test_loaders) > 1:
                print(f'================ Overall Validation Outcome ===================')
                print(f'Mean EPE across all {len(test_loaders)} sequences: {mean_EPE:.3f}')
                print('=============================================================')
        return

    assert (args.solver in ['adam', 'sgd'])
    print(f'=> setting {args.solver} solver')

    # 1. Safely extract PLIF decay parameters
    # (Ensures the SNN temporal memory parameters are isolated)
    model_obj = model.module if hasattr(model, 'module') else model
    alpha_params = [
        model_obj.alpha1, model_obj.alpha2, model_obj.alpha3, model_obj.alpha4
    ]
    alpha_param_ids = set(map(id, alpha_params))

    # 2. Categorize parameters into Refiner Head vs Coarse Backbone, and Weight vs Bias
    refiner_bias = []
    refiner_weight = []
    backbone_bias = []
    backbone_weight = []

    for name, param in model_obj.named_parameters():
        if id(param) in alpha_param_ids:
            continue
        is_refiner = ('tiny_' in name or 'convex_' in name or 'flow_inj' in name)
        is_bias = ('bias' in name)

        if is_refiner:
            if is_bias:
                refiner_bias.append(param)
            else:
                refiner_weight.append(param)
        else:
            if is_bias:
                backbone_bias.append(param)
            else:
                backbone_weight.append(param)

    # Sanity check: Ensure all parameters are accounted for
    total_grouped = len(refiner_bias) + len(refiner_weight) + len(backbone_bias) + len(backbone_weight) + len(alpha_params)
    assert total_grouped == len(list(model_obj.parameters())), (
        f"Parameter grouping mismatch: {total_grouped} grouped vs {len(list(model_obj.parameters()))} total"
    )

    snn_lr_scale = 0.01

    param_groups = []
    if refiner_bias:
        param_groups.append({'params': refiner_bias, 'lr': args.lr, 'weight_decay': 0.0, 'name': 'refiner_bias'})
    if refiner_weight:
        param_groups.append({'params': refiner_weight, 'lr': args.lr, 'weight_decay': 4e-4, 'name': 'refiner_weight'})
    if backbone_bias:
        param_groups.append({'params': backbone_bias, 'lr': args.lr * args.backbone_lr_scale, 'weight_decay': 0.0, 'name': 'backbone_bias'})
    if backbone_weight:
        param_groups.append({'params': backbone_weight, 'lr': args.lr * args.backbone_lr_scale, 'weight_decay': 4e-4, 'name': 'backbone_weight'})
    if alpha_params:
        param_groups.append({'params': alpha_params, 'lr': args.lr * snn_lr_scale, 'weight_decay': 0.0, 'name': 'alpha'})

    if is_main_process:
        print(f"=> Differential Learning Rates configured (backbone_lr_scale={args.backbone_lr_scale}):")
        print(f"   * Refiner Head: {len(refiner_weight) + len(refiner_bias)} tensors (LR: {args.lr:.6f})")
        print(f"   * Coarse Backbone: {len(backbone_weight) + len(backbone_bias)} tensors (LR: {args.lr * args.backbone_lr_scale:.6f})")
        print(f"   * PLIF Alpha: {len(alpha_params)} tensors (LR: {args.lr * snn_lr_scale:.6f})")

    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)

    # Conditional Scheduler Setup
    if args.warmup_epochs > 0:
        # Warmup for first n epochs, then multistep decay
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs
        )
        milestone_epochs = [15, 30, 45, 60, 80]
        effective_milestones = [m - args.warmup_epochs for m in milestone_epochs if m > args.warmup_epochs]
        scheduler_multistep = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=effective_milestones, gamma=0.5
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

    assert (args.train_dataset in ['mvsec', 'uzh-fpv'])

    if args.train_dataset == 'mvsec':
        train_datasets = DatasetTrain(
            args.dt,
            train_src_file,
            transform=co_transform,
            num_bins=args.num_bins
        )
        train_loader = DataLoader(
            dataset=train_datasets,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            multiprocessing_context='spawn'
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
            if is_main_process:
                print(f"Loading UZH FPV dataset {dataset_path}...")
            single_dataset = DatasetTrain(
                args.dt, 
                dataset_path, 
                transform=co_transform, 
                num_bins=args.num_bins
            )
            
            # Use DistributedSampler ONLY if in multi-GPU mode
            if is_distributed:
                from torch.utils.data.distributed import DistributedSampler
                sampler = DistributedSampler(single_dataset, shuffle=True)
                single_loader = DataLoader(
                    dataset=single_dataset, 
                    batch_size=batch_size, 
                    sampler=sampler,
                    num_workers=workers, 
                    pin_memory=True, 
                    drop_last=True
                )
            else:
                single_loader = DataLoader(
                    dataset=single_dataset, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=workers, 
                    pin_memory=True, 
                    drop_last=True,
                    multiprocessing_context='spawn'
                )
            train_loader.append(single_loader)

    # Initialize Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=False)

    for epoch in range(args.start_epoch, epochs):

        # Dynamically freeze or unfreeze coarse SNN/TPA backbone parameters for curriculum warm-start
        freeze_bb = (epoch < args.freeze_backbone_epochs)
        model_obj = model.module if hasattr(model, 'module') else model
        for name, param in model_obj.named_parameters():
            if 'tiny_' not in name and 'convex_' not in name and 'flow_inj' not in name:
                param.requires_grad = not freeze_bb
        
        if is_main_process:
            if freeze_bb:
                print(f"=> [Curriculum] Epoch {epoch}: Coarse backbone FROZEN (training Tiny Refiner & Convex Head only).")
            elif epoch == args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0:
                print(f"=> [Curriculum] Epoch {epoch}: Coarse backbone UNFROZEN (full end-to-end training active with backbone_lr_scale={args.backbone_lr_scale}).")

        refiner_lr = optimizer.param_groups[0]['lr']
        backbone_lr = refiner_lr
        for pg in optimizer.param_groups:
            if pg.get('name') == 'backbone_weight':
                backbone_lr = pg['lr']
                break
        print(f"Learning Rate: {refiner_lr:.6f} (Backbone LR: {backbone_lr:.6f})")

        if args.train_dataset in ['uzh-fpv']:
            # Shuffle the order we read the HDF5 sequence files every epoch
            random.shuffle(train_loader)
            
            total_loss_sum = 0.0
            total_samples = 0
            for loader in train_loader:
                # Train fully on one file before moving to the next
                loss = train(loader, model, optimizer, epoch, train_writer, scaler)
                num_samples = len(loader.dataset)
                total_loss_sum += loss * num_samples
                total_samples += num_samples
            
            train_loss = total_loss_sum / max(total_samples, 1)
        else:
            # Standard MVSEC single-loader logic
            train_loss = train(train_loader, model, optimizer, epoch, train_writer, scaler)

        train_writer.add_scalar('mean_train_loss', train_loss, epoch)

        print(f"Mean Training Loss: {train_loss:.3f} of epoch {epoch}")
        print('-------------------------------------------------------')

        scheduler.step()

        # Set sampler epoch in the training loop
        if is_distributed and args.train_dataset == 'uzh-fpv':
            for loader in train_loader:
                if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(epoch)

        # Test at every n epoch during training
        if (epoch + 1) % args.eval_int == 0:
            # evaluate on validation set
            with torch.no_grad():
                total_EPE = 0
                eval_net = model.module if hasattr(model, 'module') else model
                for t_loader, t_src, t_gt, t_env in test_loaders:
                    total_EPE += validate(t_loader, eval_net, epoch, output_writers if is_main_process else [], t_src, t_gt, t_env)
                EPE = total_EPE / len(test_loaders)
            
            if is_main_process:
                if len(test_loaders) > 1:
                    print(f'================ Overall Validation Outcome (Epoch {epoch}) ===================')
                    print(f'Mean EPE across all {len(test_loaders)} sequences: {EPE:.3f}')
                    print('=============================================================================')

                test_writer.add_scalar('mean_val_EPE', EPE, epoch)

            if best_EPE < 0:
                best_EPE = EPE
                is_best = True
            else:
                is_best = EPE < best_EPE
                best_EPE = min(EPE, best_EPE)

            if is_main_process:
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'arch': arch,
                    'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'best_EPE': best_EPE,
                    'div_flow': div_flow,
                    'num_bins': args.num_bins
                }

                # 1. ALWAYS save model_best.pth.tar whenever a new record EPE is achieved!
                if is_best:
                    save_checkpoint(checkpoint_data, True, save_path, filename=f'checkpoint_best_{EPE:.4f}.pth.tar')
                    print(f"=> [CHECKPOINT] New BEST model achieved (EPE: {EPE:.4f})! Saved to {os.path.join(save_path, 'model_best.pth.tar')}")

                # 2. Archive intermediate epoch checkpoints if EPE is below save_thred
                if EPE < args.save_thred:
                    filename = f'checkpoint_epoch_{epoch + 1}_{EPE:.4f}.pth.tar'
                    save_checkpoint(checkpoint_data, False, save_path, filename=filename)
                    print(f"=> [CHECKPOINT] Validation EPE {EPE:.4f} < save_thred {args.save_thred}. Saved {filename}")

            # check if exit criteria is met
            if is_best:
                val_fail_times = 0
            else:
                val_fail_times += 1

            if val_fail_times >= args.max_fail_times:
                if is_main_process:
                    print(
                        "Epoch {}: validation failed for consective {} times".format(
                            epoch, val_fail_times
                        )
                    )
                break

            if is_distributed:
                dist.barrier()


if __name__ == '__main__':
    main()
