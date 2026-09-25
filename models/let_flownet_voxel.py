import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import constant_
from einops import rearrange

from .base_model import BaseModel
from .snn.spiking_lif import LIF_Neuron
from .trans.transformer_encoder import transformer_encoder
from .trans.transformer_decoder import transformer_decoder
from .trans.position_encoding import build_position_encoding
from .model_util import ConvLayer, UpsampleConvLayer, conv_s, deconv, predict_flow


__all__ = ['let_flownet_voxel']


class Let_Flownet_Voxel(BaseModel):
    expansion = 1

    def __init__(self, args, device, batchNorm=True):
        super(Let_Flownet_Voxel, self).__init__()

        self.args = args
        self.device = device
        dt = getattr(args, 'dt', 1)

        # SNN
        self.batchNorm = batchNorm
        self.conv_s1 = conv_s(self.batchNorm, 2, 64, stride=2)
        self.conv_s2 = conv_s(self.batchNorm, 64, 128, stride=2)
        self.conv_s3 = conv_s(self.batchNorm, 128, 256, stride=2)
        self.conv_s4 = conv_s(self.batchNorm, 256, 512, stride=2)

        self.inh_s1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False)
        self.inh_s2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.inh_s3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False)
        self.inh_s4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)

        # use 3 for dt1, and 2 for dt4 and larger (e.g. dt8)
        numerator = 3.0 if dt == 1 else 2.0
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(numerator / n)
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

        # Divide the total window time by the number of bins to get the time per step
        time_step = (dt * 10 * 1e-3) / self.args.num_bins

        # Initialize separate learnable alpha parameters for each channel depth
        # Calculate target decay
        target_alpha = np.exp(-time_step / self.args.tau)
        # Convert to logit for the sigmoid in the forward pass
        init_logit = math.log(target_alpha / (1.0 - target_alpha))

        self.alpha1 = torch.nn.Parameter(torch.full((1, 64, 1, 1), init_logit, dtype=torch.float32))
        self.alpha2 = torch.nn.Parameter(torch.full((1, 128, 1, 1), init_logit, dtype=torch.float32))
        self.alpha3 = torch.nn.Parameter(torch.full((1, 256, 1, 1), init_logit, dtype=torch.float32))
        self.alpha4 = torch.nn.Parameter(torch.full((1, 512, 1, 1), init_logit, dtype=torch.float32))

        # Transformers
        norm = self.args.norm

        self.position_embedding = build_position_encoding('sine', 512)
        self.split0 = nn.Unfold(kernel_size=1, stride=1, padding=0)

        num_enc_layers = self.args.num_enc_layers
        num_dec_layers = self.args.num_dec_layers

        # TODO: 1. try different dropout
        self.trans_encoder0 = transformer_encoder(d_model=512, nhead=8, num_encoder_layers=num_enc_layers,
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        self.trans_decoder0 = transformer_decoder(d_model=512, nhead=8, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        
        self.split1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.trans_encoder1 = transformer_encoder(d_model=512, nhead=8, num_encoder_layers=num_enc_layers, 
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        self.trans_decoder1 = transformer_decoder(d_model=512, nhead=8, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        
        self.split2 = nn.Sequential(
            ConvLayer(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, norm=norm),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        )
        self.trans_encoder2 = transformer_encoder(d_model=512, nhead=8, num_encoder_layers=num_enc_layers, 
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        self.trans_decoder2 = transformer_decoder(d_model=512, nhead=8, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        
        self.split3 = nn.Sequential(
            ConvLayer(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, norm=norm),
            ConvLayer(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, norm=norm),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        )
        self.trans_encoder3 = transformer_encoder(d_model=512, nhead=8, num_encoder_layers=num_enc_layers, 
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        self.trans_decoder3 = transformer_decoder(d_model=512, nhead=8, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=1024, activation='relu', dropout=0.1)

        self.deconv = nn.ModuleList([
            deconv(self.batchNorm, 512, 128),
            deconv(self.batchNorm, 1024, 128),
            deconv(self.batchNorm, 832, 128)
        ])
        
        self.UpsampleConv = nn.ModuleList([
            UpsampleConvLayer(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2, norm=norm),
            UpsampleConvLayer(in_channels=1024, out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm),
            UpsampleConvLayer(in_channels=832, out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm),
            UpsampleConvLayer(in_channels=768, out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm)
        ])

        self.predict_flow = nn.ModuleList([
            predict_flow(self.batchNorm, 256, 128),
            predict_flow(self.batchNorm, 128, 64),
            predict_flow(self.batchNorm, 128, 64),
            predict_flow(self.batchNorm, 128, 3)
        ])

        # ---> NEW: True 3-Channel Flow Projectors for Multi-Scale Loss <---
        self.flow_projectors = nn.ModuleList([
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        ])
        
        # Initialize with near-zero weights to prevent chaotic warping in Epoch 0
        for m in self.flow_projectors:
            nn.init.normal_(m.weight, 0, 0.0001)
            nn.init.constant_(m.bias, 0)

    def encode_snn(self, input, sp_threshold):
        B, _, H, W, num_bins = input.size()
        threshold = sp_threshold

        mem_1 = torch.zeros(B, 64, H // 2, W // 2, device=input.device)
        mem_2 = torch.zeros(B, 128, H // 4, W // 4, device=input.device)
        mem_3 = torch.zeros(B, 256, H // 8, W // 8, device=input.device)
        mem_4 = torch.zeros(B, 512, H // 16, W // 16, device=input.device)

        mem_1_total = torch.zeros_like(mem_1)
        mem_2_total = torch.zeros_like(mem_2)
        mem_3_total = torch.zeros_like(mem_3)
        mem_4_total = torch.zeros_like(mem_4)

        for i in range(num_bins):
            input11 = input[:, :, :, :, i]

            current_1 = self.conv_s1(input11)
            G_inh_1 = torch.sigmoid(self.inh_s1(current_1))
            mem_1 = (torch.sigmoid(self.alpha1) * mem_1 + current_1) * (1 - G_inh_1)
            mem_1, spike_1 = LIF_Neuron(mem_1, threshold)
            mem_1_total = mem_1_total + current_1
            if i == 0:
                mem_1_start = mem_1.clone()
            if i == num_bins - 1:
                mem_1_end = mem_1.clone()

            current_2 = self.conv_s2(spike_1)
            G_inh_2 = torch.sigmoid(self.inh_s2(current_2))
            mem_2 = (torch.sigmoid(self.alpha2) * mem_2 + current_2) * (1 - G_inh_2)
            mem_2, spike_2 = LIF_Neuron(mem_2, threshold)
            mem_2_total = mem_2_total + current_2
            if i == 0:
                mem_2_start = mem_2.clone()
            if i == num_bins - 1:
                mem_2_end = mem_2.clone()

            current_3 = self.conv_s3(spike_2)
            G_inh_3 = torch.sigmoid(self.inh_s3(current_3))
            mem_3 = (torch.sigmoid(self.alpha3) * mem_3 + current_3) * (1 - G_inh_3)
            mem_3, spike_3 = LIF_Neuron(mem_3, threshold)
            mem_3_total = mem_3_total + current_3
            if i == 0:
                mem_3_start = mem_3.clone()
            if i == num_bins - 1:
                mem_3_end = mem_3.clone()

            current_4 = self.conv_s4(spike_3)
            G_inh_4 = torch.sigmoid(self.inh_s4(current_4))
            mem_4 = (torch.sigmoid(self.alpha4) * mem_4 + current_4) * (1 - G_inh_4)
            mem_4, spike_4 = LIF_Neuron(mem_4, threshold)
            mem_4_total = mem_4_total + current_4
            if i == 0:
                mem_4_start = mem_4.clone()
            if i == num_bins - 1:
                mem_4_end = mem_4.clone()

        blocks = [
            mem_1_total + (mem_1_end - mem_1_start),
            mem_2_total + (mem_2_end - mem_2_start),
            mem_3_total + (mem_3_end - mem_3_start),
            mem_4_total + (mem_4_end - mem_4_start)
        ]
        return blocks

    def decode_flow(self, blocks, H, W):
        # TPA Transformer Paths
        token0 = self.split0(blocks[-1]).transpose(1, 2)
        pos0 = self.position_embedding(token0)
        hs0 = self.trans_encoder0(src=token0.transpose(0, 1), pos=pos0.transpose(0, 1))
        hc0 = self.trans_decoder0(tgt=hs0, memory=hs0)

        token1 = self.split1(blocks[-2]).flatten(2).transpose(1, 2)
        pos1 = self.position_embedding(token1)
        hs1 = self.trans_encoder1(src=token1.transpose(0, 1), pos=pos1.transpose(0, 1))
        hc1 = self.trans_decoder1(tgt=hs1, memory=hs0)

        token2 = self.split2(blocks[-3]).flatten(2).transpose(1, 2)
        pos2 = self.position_embedding(token2)
        hs2 = self.trans_encoder2(src=token2.transpose(0, 1), pos=pos2.transpose(0, 1))
        hc2 = self.trans_decoder2(tgt=hs2, memory=hs1)

        token3 = self.split3(blocks[-4]).flatten(2).transpose(1, 2)
        pos3 = self.position_embedding(token3)
        hs3 = self.trans_encoder3(src=token3.transpose(0, 1), pos=pos3.transpose(0, 1))
        hc3 = self.trans_decoder3(tgt=hs3, memory=hs2)

        hc0_img = rearrange(hc0, '(h w) n c -> n c h w', h=H//16, w=W//16)
        hc1_img = rearrange(hc1, '(h w) n c -> n c h w', h=H//16, w=W//16)
        hc2_img = rearrange(hc2, '(h w) n c -> n c h w', h=H//16, w=W//16)
        hc3_img = rearrange(hc3, '(h w) n c -> n c h w', h=H//16, w=W//16)

        # Multi-Level Upsampler (MLU) Decoder
        input0 = self.UpsampleConv[0](hc0_img)
        flow0_feat = self.predict_flow[0](input0)
        true_flow0 = self.flow_projectors[0](flow0_feat)
        hs_up = self.deconv[0](hc0_img)

        hc1_up = F.interpolate(hc1_img, scale_factor=2, mode='bilinear', align_corners=False)
        concat1 = torch.cat((flow0_feat, blocks[2], hs_up, hc1_up), 1)
        input1 = self.UpsampleConv[1](concat1)
        flow1_feat = self.predict_flow[1](input1)
        true_flow1 = self.flow_projectors[1](flow1_feat)
        concat1_up = self.deconv[1](concat1)

        hc2_up = F.interpolate(hc2_img, scale_factor=4, mode='bilinear', align_corners=False)
        concat2 = torch.cat((flow1_feat, blocks[1], concat1_up, hc2_up), 1)
        input2 = self.UpsampleConv[2](concat2)
        flow2_feat = self.predict_flow[2](input2)
        true_flow2 = self.flow_projectors[2](flow2_feat)
        concat2_up = self.deconv[2](concat2)

        hc3_up = F.interpolate(hc3_img, scale_factor=8, mode='bilinear', align_corners=False)
        concat3 = torch.cat((flow2_feat, blocks[0], concat2_up, hc3_up), 1)
        input3 = self.UpsampleConv[3](concat3)
        flow3 = self.predict_flow[3](input3)

        return [true_flow0, true_flow1, true_flow2, flow3]

    def warp_features(self, x, flow):
        """
        Warps feature tensor x [B, C, H, W] using flow [B, 2, H, W] in pixel coordinates.
        """
        B, C, H, W = x.size()
        # Create coordinate grid in pixel space [B, H, W, 2]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1) # [B, H, W, 2]

        # Flow is [B, 2, H, W] -> permute to [B, H, W, 2]
        flow_perm = flow.permute(0, 2, 3, 1)
        target_grid = grid + flow_perm

        # Normalize coordinates to [-1, 1] for F.grid_sample
        target_grid[..., 0] = 2.0 * target_grid[..., 0] / max(W - 1, 1) - 1.0
        target_grid[..., 1] = 2.0 * target_grid[..., 1] / max(H - 1, 1) - 1.0

        return F.grid_sample(x, target_grid, mode='bilinear', padding_mode='border', align_corners=True)

    def forward(self, input, sp_threshold, iters=2):
        B, _, H, W, num_bins = input.size()

        # Step 1: SNN extracts temporal features once
        blocks = self.encode_snn(input, sp_threshold)

        # Step 2: Pass 1 (Coarse Estimation)
        flows_p1 = self.decode_flow(blocks, H, W)
        
        if iters == 1:
            return flows_p1

        # Step 3: Feature Warping via grid_sample
        # flows_p1[-1] (flow3) has shape [B, 2 or 3, H//2, W//2]
        # Slice :2 to isolate 2D translation vectors (u, v)
        coarse_flow = flows_p1[-1][:, :2].detach()

        # Scale flow magnitude and resize for each specific SNN block dimension
        flow_b0 = F.interpolate(coarse_flow * 0.5, size=(H//2, W//2), mode='bilinear', align_corners=False)
        flow_b1 = F.interpolate(coarse_flow * 0.25, size=(H//4, W//4), mode='bilinear', align_corners=False)
        flow_b2 = F.interpolate(coarse_flow * 0.125, size=(H//8, W//8), mode='bilinear', align_corners=False)
        flow_b3 = F.interpolate(coarse_flow * 0.0625, size=(H//16, W//16), mode='bilinear', align_corners=False)

        # Warp all 4 scale blocks using their perfectly matched flow grids
        warped_blocks = [
            self.warp_features(blocks[0], flow_b0),
            self.warp_features(blocks[1], flow_b1),
            self.warp_features(blocks[2], flow_b2),
            self.warp_features(blocks[3], flow_b3)
        ]

        # Step 4: Pass 2 (Fine Residual Estimation)
        flows_res = self.decode_flow(warped_blocks, H, W)

        # Step 5: Add coarse flow to the predicted residual
        final_flows = [
            flows_p1[0] + flows_res[0],
            flows_p1[1] + flows_res[1],
            flows_p1[2] + flows_res[2],
            flows_p1[3] + flows_res[3]
        ]

        # Returns multi-scale outputs for loss supervision
        return final_flows

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def let_flownet_voxel(args, device, data=None):
    model = Let_Flownet_Voxel(args, device)

    if data is not None:
        try:
            model.load_state_dict(data['state_dict'], strict=False)
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")
            print("Continuing without pre-trained weights.")

    return model
