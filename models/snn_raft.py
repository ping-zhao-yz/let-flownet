import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import constant_

from .base_model import BaseModel
from .snn.spiking_lif import LIF_Neuron
from .raft_core.update import BasicUpdateBlock
from .raft_core.corr import CorrBlock
from .raft_core.utils import coords_grid, upsample_flow

class SNNEncoder(nn.Module):
    def __init__(self, args, batchNorm=True):
        super(SNNEncoder, self).__init__()
        self.args = args
        dt = getattr(args, 'dt', 1)

        from .model_util import conv_s
        self.conv_s1 = conv_s(batchNorm, 2, 64, stride=2)
        self.conv_s2 = conv_s(batchNorm, 64, 128, stride=2)
        self.conv_s3 = conv_s(batchNorm, 128, 256, stride=2)

        self.inh_s1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.inh_s2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.inh_s3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False)

        numerator = 3.0 if dt == 1 else 2.0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(numerator / n)
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

        # Since we sliced the temporal input into 5 bins per encoder
        num_bins_half = self.args.num_bins // 2
        time_step = (dt * 10 * 1e-3) / self.args.num_bins

        target_alpha = np.exp(-time_step / self.args.tau)
        init_logit = math.log(target_alpha / (1.0 - target_alpha))

        self.alpha1 = torch.nn.Parameter(torch.full((1, 64, 1, 1), init_logit, dtype=torch.float32))
        self.alpha2 = torch.nn.Parameter(torch.full((1, 128, 1, 1), init_logit, dtype=torch.float32))
        self.alpha3 = torch.nn.Parameter(torch.full((1, 256, 1, 1), init_logit, dtype=torch.float32))
        self.num_bins_half = num_bins_half

    def forward(self, input_tensor, threshold):
        # input_tensor is [B, 2, H, W, num_bins_half]
        B, C, H, W, _ = input_tensor.size()

        mem_1 = torch.zeros(B, 64, H // 2, W // 2).to(input_tensor.device)
        mem_2 = torch.zeros(B, 128, H // 4, W // 4).to(input_tensor.device)
        mem_3 = torch.zeros(B, 256, H // 8, W // 8).to(input_tensor.device)

        mem_1_total = torch.zeros_like(mem_1)
        mem_2_total = torch.zeros_like(mem_2)
        mem_3_total = torch.zeros_like(mem_3)

        spike_1 = torch.zeros_like(mem_1)
        spike_2 = torch.zeros_like(mem_2)
        spike_3 = torch.zeros_like(mem_3)

        for i in range(self.num_bins_half):
            input11 = input_tensor[:, :, :, :, i]

            current_1 = self.conv_s1(input11)
            G_inh_1 = torch.sigmoid(self.inh_s1(current_1))
            mem_1 = (torch.sigmoid(self.alpha1) * mem_1 + current_1) * (1 - G_inh_1)
            mem_1, spike_1 = LIF_Neuron(mem_1, threshold)
            mem_1_total = mem_1_total + current_1
            if i == 0:
                mem_1_start = mem_1.clone()
            if i == self.num_bins_half - 1:
                mem_1_end = mem_1.clone()

            current_2 = self.conv_s2(spike_1)
            G_inh_2 = torch.sigmoid(self.inh_s2(current_2))
            mem_2 = (torch.sigmoid(self.alpha2) * mem_2 + current_2) * (1 - G_inh_2)
            mem_2, spike_2 = LIF_Neuron(mem_2, threshold)
            mem_2_total = mem_2_total + current_2
            if i == 0:
                mem_2_start = mem_2.clone()
            if i == self.num_bins_half - 1:
                mem_2_end = mem_2.clone()

            current_3 = self.conv_s3(spike_2)
            G_inh_3 = torch.sigmoid(self.inh_s3(current_3))
            mem_3 = (torch.sigmoid(self.alpha3) * mem_3 + current_3) * (1 - G_inh_3)
            mem_3, spike_3 = LIF_Neuron(mem_3, threshold)
            mem_3_total = mem_3_total + current_3
            if i == 0:
                mem_3_start = mem_3.clone()
            if i == self.num_bins_half - 1:
                mem_3_end = mem_3.clone()

        out = mem_3_total + (mem_3_end - mem_3_start)
        return out


class SNN_RAFT(BaseModel):
    def __init__(self, args, device, batchNorm=True):
        super(SNN_RAFT, self).__init__()
        self.args = args
        self.device = device

        self.fnet = SNNEncoder(args, batchNorm=batchNorm)
        self.cnet = SNNEncoder(args, batchNorm=batchNorm)
        
        # Add 1x1 convs to map cnet 256 channels to (net=128, inp=128)
        self.cnet_proj = nn.Conv2d(256, 256, kernel_size=1)

        class RAFTArgs:
            pass
        raft_args = RAFTArgs()
        raft_args.corr_levels = 4
        raft_args.corr_radius = 4
        raft_args.mixed_precision = False
        
        self.update_block = BasicUpdateBlock(raft_args, hidden_dim=128)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        return coords0, coords1

    def forward(self, input, sp_threshold, iters=12, test_mode=False):
        B, C, H, W, num_bins = input.size()

        # Split temporally
        mid = num_bins // 2
        input_t0 = input[:, :, :, :, :mid]
        input_t1 = input[:, :, :, :, mid:]

        # Feature encoder
        fmap1 = self.fnet(input_t0, sp_threshold)
        fmap2 = self.fnet(input_t1, sp_threshold)
        
        # Context encoder
        cnet = self.cnet(input_t0, sp_threshold)
        cnet = self.cnet_proj(cnet)
        net, inp = torch.split(cnet, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        corr_fn = CorrBlock(fmap1, fmap2, num_levels=4, radius=4)

        coords0, coords1 = self.initialize_flow(input[:, :, :, :, 0])
        
        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) 
            
            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            
            coords1 = coords1 + delta_flow
            
            # ---> Apply Convex Upsampling <---
            flow_up = upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        if test_mode:
            return flow_predictions[-1]

        return flow_predictions


def snn_raft(args, device, data=None):
    model = SNN_RAFT(args, device)
    if data is not None:
        try:
            model.load_state_dict(data['state_dict'], strict=False)
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")
            print("Continuing without pre-trained weights.")
    return model
