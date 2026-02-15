import argparse
import cv2
import h5py
import numpy as np
import os
import os.path
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from util.loss_util import AverageMeter
from util.flow_util import flow2rgb, flow_viz_np, save_checkpoint

from datasets.evt_count_divided.dataset_dtx import DatasetTest, DatasetTrain
from models import spiket_flownet_snn_lif_trans
from loss.multiscaleloss import estimate_corresponding_gt_flow, flow_error_dense, smooth_loss_upsample_single
from loss.photometric_loss_backward import photometric_loss_backward_single


parser = argparse.ArgumentParser(description='spiket_flownet_snn_lif_trans training on several datasets',
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

parser.add_argument('--num_enc_layers', default=4, help='number of transformer encoder layers')
parser.add_argument('--num_dec_layers', default=4, help='number of transformer decoder layers')

parser.add_argument('--mixed_precision', action='store_true',
                    help='use mixed precision')

parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--dt', type=int, default=4, help='time interval (1, 4, or 8)')
parser.add_argument('--sp_threshold', type=float, default=0.5, help='spike threshold')

args = parser.parse_args()

# Initializations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=> using device '{device}'")

image_resize = 256
sp_threshold = args.sp_threshold

div_flow = 1

dataset_dir = '../../../dataset/Event/mvsec/preprocessed'
src_file_dir = '../../../dataset/Event/mvsec/original'

save_dir = 'spiket_flownet_snn_lif_trans_dt4_output'

train_env = 'outdoor_day2'
test_env = 'indoor_flying1'

train_dir = os.path.join(dataset_dir, train_env)
test_dir = os.path.join(dataset_dir, test_env)

train_src_file = src_file_dir + '/' + train_env + '/' + train_env + "_data.hdf5"
test_src_file = src_file_dir + '/' + test_env + '/' + test_env + "_data.hdf5"
test_gt_file = src_file_dir + '/' + test_env + '/' + test_env + "_gt.hdf5"

arch = "spiket_flownet_snn_lif_trans"

lr = 5e-5
epochs = 100
batch_size = 8
iter_g = 0


def train(train_loader, model, optimizer, epoch, train_writer):
    global iter_g, args, image_resize, sp_threshold
    np.set_printoptions(precision=2)
    losses = AverageMeter()

    # switch to train mode
    model.train()

    multiscale_weights = [1, 1, 1, 1]
    print_freq = 100

    for i_batch, data in enumerate(train_loader, 0):
        # get the inputs
        former_inputs_on, former_inputs_off, latter_inputs_on, latter_inputs_off, former_gray, latter_gray = data

        if torch.sum(former_inputs_on + former_inputs_off) > 0:
            print_details = i_batch % print_freq == 0

            if i_batch == 0:
                print(f"VERIFICATION: dt={args.dt}, N (event frames) = {former_inputs_on.shape[-1]}")

            event_data = initInputRepresentation(
                former_inputs_on, former_inputs_off, latter_inputs_on, latter_inputs_off)

            # compute output
            flow_predictions = model(event_data, image_resize, sp_threshold)

            # Photometric loss.
            photometric_loss = photometric_loss_backward_single(former_gray[:, 0, :, :], latter_gray[:, 0, :, :], torch.sum(
                event_data, 4), flow_predictions, device, print_details, weights=multiscale_weights)

            # Smoothness loss.
            smoothness_loss = smooth_loss_upsample_single(flow_predictions)

            # total_loss
            loss = photometric_loss + 10 * smoothness_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss and EPE
            train_writer.add_scalar('train_loss', loss.item(), iter_g)
            losses.update(loss.item(), event_data.size(0))

            if print_details:
                now = datetime.strftime(datetime.now(), "%d-%m-%Y_%H-%M-%S")
                print(f'Time: {now}, Epoch: [{epoch}][{batch_size * i_batch}/{batch_size * len(train_loader)}], Loss: {losses}, photometric_loss: {round(photometric_loss.item(), 2)}, smoothness_loss: {smoothness_loss.item():.2f}')
                print('-------------------------------------------------------')

            iter_g += 1

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
    percent_Outlier_sum = 0.
    iters = 0.
    scale = 1

    print_freq = 100

    for i_batch, data in enumerate(test_loader, 0):
        former_inputs_on, former_inputs_off, latter_inputs_on, latter_inputs_off, ts_f, ts_l = data

        if torch.sum(former_inputs_on + former_inputs_off) > 0:
            event_data = initInputRepresentation(
                former_inputs_on, former_inputs_off, latter_inputs_on, latter_inputs_off)

            # compute output
            output = model(event_data, image_resize, sp_threshold)
            output_temp = output.cpu()

            pred_flow = np.zeros((image_resize, image_resize, 2), dtype=np.float32)
            pred_flow[:, :, 0] = cv2.resize(np.array(
                output_temp[0, 0, :, :]), (image_resize, image_resize), interpolation=cv2.INTER_LINEAR)
            pred_flow[:, :, 1] = cv2.resize(np.array(
                output_temp[0, 1, :, :]), (image_resize, image_resize), interpolation=cv2.INTER_LINEAR)

            u_gt_all = np.array(gt_temp[:, 0, :, :])
            v_gt_all = np.array(gt_temp[:, 1, :, :])

            u_gt, v_gt = estimate_corresponding_gt_flow(
                u_gt_all, v_gt_all, gt_ts_temp, np.array(ts_f), np.array(ts_l))
            gt_flow = np.stack((u_gt, v_gt), axis=2)

            #   ----------- Visualization
            if epoch < 0 and not torch.cuda.is_available():
                mask_temp = former_inputs_on + former_inputs_off + \
                    latter_inputs_on + latter_inputs_off
                mask_temp = torch.sum(torch.sum(mask_temp, 0), 2)
                mask_temp_np = np.squeeze(np.array(mask_temp)) > 0

                spike_image = mask_temp
                spike_image[spike_image > 0] = 255
                cv2.imshow('Spike Image', np.array(
                    spike_image, dtype=np.uint8))

                gray = cv2.resize(
                    gray_image[i_batch], (scale*image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Gray Image', cv2.cvtColor(
                    gray, cv2.COLOR_BGR2RGB))

                out_temp = np.array(output_temp.cpu().detach())
                x_flow = cv2.resize(np.array(out_temp[0, 0, :, :]), (
                    scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                y_flow = cv2.resize(np.array(out_temp[0, 1, :, :]), (
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

                x_flow_masked = cv2.resize(np.array(out_temp[0, 0, :, :] * mask_temp_np), (
                    scale * image_resize, scale * image_resize), interpolation=cv2.INTER_LINEAR)
                y_flow_masked = cv2.resize(np.array(out_temp[0, 1, :, :] * mask_temp_np), (
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

            gt_flow = gt_flow[yoff: -yoff if yoff > 0 else None, xoff: -xoff if xoff > 0 else None, :]

            AEE, percent_Outlier, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = flow_error_dense(
                gt_flow, pred_flow, (torch.sum(torch.sum(torch.sum(event_data, dim=0), dim=0), dim=2)).cpu(), is_car=False)

            AEE_sum = AEE_sum + div_flow * AEE
            AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

            AEE_sum_gt = AEE_sum_gt + div_flow * AEE_gt
            AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

            percent_Outlier_sum += percent_Outlier

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
    print('Mean AEE: {:.3f}, sum AEE: {:.2f}, Mean AEE_gt: {:.2f}, sum AEE_gt: {:.2f}, Mean %Outlier: {:.3f}, # pts: {:.2f}'
        .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters, percent_Outlier_sum / iters, n_points))
    print('===============================================================')

    gt_temp = None

    return AEE_sum / iters


def initInputRepresentation(former_inputs_on, former_inputs_off, latter_inputs_on, latter_inputs_off):

    input_representation = torch.zeros(
        former_inputs_on.size(0), 4, image_resize, image_resize, former_inputs_on.size(3)).float()

    for b in range(4):
        if b == 0:
            input_representation[:, 0, :, :, :] = former_inputs_on
        elif b == 1:
            input_representation[:, 1, :, :, :] = former_inputs_off
        elif b == 2:
            input_representation[:, 2, :, :, :] = latter_inputs_on
        elif b == 3:
            input_representation[:, 3, :, :, :] = latter_inputs_off

    return input_representation.type(torch.FloatTensor).to(device)


def main():
    global args

    workers = 4
    best_EPE = -1
    evaluate_interval = 5

    val_fail_times_max = 5
    val_fail_times = 0

    save_path = '{},{},epochs{},bat{},lr{}'.format(
        arch,
        args.solver,
        epochs,
        batch_size,
        lr)

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

    Test_dataset = DatasetTest(args.dt, test_src_file, test_dir)
    test_loader = DataLoader(dataset=Test_dataset,
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

    model = spiket_flownet_snn_lif_trans.__dict__[arch](args, device, network_data).to(device)
    model = torch.nn.DataParallel(model).to(device)
    cudnn.benchmark = True

    if args.evaluate:
        with torch.no_grad():
            best_EPE = validate(test_loader, model, -1, output_writers)
        return

    assert (args.solver in ['adam', 'sgd'])
    print(f'=> setting {args.solver} solver')
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': 0},
                    {'params': model.module.weight_parameters(), 'weight_decay': 4e-4}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(
            param_groups, lr, betas=(0.9, 0.999))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups, lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [5, 10, 20, 30, 40, 50, 60, 70, 80, 90], gamma=0.7)

    co_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomCrop((256, 256)),
        transforms.ToTensor(),
    ])

    Train_dataset = DatasetTrain(
        args.dt, train_src_file, train_dir, transform=co_transform)
    train_loader = DataLoader(dataset=Train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers)

    for epoch in range(args.start_epoch, epochs):
        train_loss = train(train_loader, model, optimizer, epoch, train_writer)
        train_writer.add_scalar('mean_train_loss', train_loss, epoch)

        scheduler.step()

        # Test at every 5 epoch during training
        if (epoch + 1) % evaluate_interval == 0:
            # evaluate on validation set
            with torch.no_grad():
                EPE = validate(test_loader, model, epoch, output_writers)
            test_writer.add_scalar('mean_val_EPE', EPE, epoch)

            if best_EPE < 0:
                best_EPE = EPE

            # check if exit criteria is met
            if EPE < best_EPE:
                val_fail_times = 0
            else:
                val_fail_times += 1

            if val_fail_times >= val_fail_times_max:
                print(
                    "Epoch {}: validation failed for consective {} times".format(
                        epoch, val_fail_times
                    )
                )
                break

            is_best = EPE < best_EPE
            best_EPE = min(EPE, best_EPE)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.module.state_dict(),
                'best_EPE': best_EPE,
                'div_flow': div_flow
            }, is_best, save_path)


if __name__ == '__main__':
    main()
