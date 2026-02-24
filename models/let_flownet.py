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

from torch.utils.checkpoint import checkpoint

__all__ = ['let_flownet']


class LET_FlowNet(BaseModel):
    expansion = 1

    def __init__(self, args, device, batchNorm=True):
        super(LET_FlowNet, self).__init__()

        self.args = args
        self.device = device
        dt = getattr(args, 'dt', 1)
        norm = self.args.norm

        # 1. Pyramidal SNN Encoder
        self.batchNorm = batchNorm
        self.conv_s1 = conv_s(self.batchNorm, 4, 64, stride=2)
        self.conv_s2 = conv_s(self.batchNorm, 64, 128, stride=2)
        self.conv_s3 = conv_s(self.batchNorm, 128, 256, stride=2)
        self.conv_s4 = conv_s(self.batchNorm, 256, 512, stride=2)

        # Weight initialization
        numerator = 3.0 if dt == 1 else 2.0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(numerator / n)
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

        time_step = dt * 10 * 1e-3
        self.alpha = np.exp(-time_step/self.args.tau)

        # 2. TPA (Transformer Pyramid): d_model=256, dim_ff=512
        num_enc_layers = self.args.num_enc_layers
        num_dec_layers = self.args.num_dec_layers
        d_model = 256
        dim_ff = 512
        
        # 4096 supports 64x64 tokens (Scale 2 and 3)
        self.position_embedding = build_position_encoding('sine', d_model, n_position=4096)
        
        # Scale 0-3 Tokenizers: Map various SNN outputs to d_model=256
        self.split0 = ConvLayer(512, d_model, kernel_size=3, stride=1, padding=1, norm=norm)
        self.trans_encoder0 = transformer_encoder(d_model=d_model, nhead=8, num_encoder_layers=num_enc_layers,
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        self.trans_decoder0 = transformer_decoder(d_model=d_model, nhead=8, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        
        self.split1 = ConvLayer(256, d_model, kernel_size=3, stride=1, padding=1, norm=norm)
        self.trans_encoder1 = transformer_encoder(d_model=d_model, nhead=8, num_encoder_layers=num_enc_layers, 
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        self.trans_decoder1 = transformer_decoder(d_model=d_model, nhead=8, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        
        self.split2 = ConvLayer(128, d_model, kernel_size=3, stride=1, padding=1, norm=norm)
        self.trans_encoder2 = transformer_encoder(d_model=d_model, nhead=8, num_encoder_layers=num_enc_layers, 
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        self.trans_decoder2 = transformer_decoder(d_model=d_model, nhead=8, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        
        # Scale 3: Stride 2 projects 128x128 -> 64x64 tokens to avoid OOM
        self.split3 = ConvLayer(64, d_model, kernel_size=3, stride=2, padding=1, norm=norm)
        self.trans_encoder3 = transformer_encoder(d_model=d_model, nhead=8, num_encoder_layers=num_enc_layers, 
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        self.trans_decoder3 = transformer_decoder(d_model=d_model, nhead=8, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)

        # 3. MLU (Multi-Level Upsampler): Hierarchical Fusion
        # Stage 1: flow(2) + snn_m3(256) + deconv(128) + hc1(256) = 642
        # Stage 2: flow(2) + snn_m2(128) + deconv(128) + hc2(256) = 514
        # Stage 3: flow(2) + snn_m1(64) + deconv(128) + hc3(256) = 450
        self.deconv = nn.ModuleList([
            deconv(self.batchNorm, d_model, 128), # Stage 0
            deconv(self.batchNorm, 642, 128),     # Stage 1
            deconv(self.batchNorm, 514, 128)      # Stage 2
        ])
        
        self.UpsampleConv = nn.ModuleList([
            UpsampleConvLayer(in_channels=d_model, out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm),
            UpsampleConvLayer(in_channels=642,     out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm),
            UpsampleConvLayer(in_channels=514,     out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm),
            UpsampleConvLayer(in_channels=450,     out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm)
        ])

        self.predict_flow = nn.ModuleList([
            predict_flow(self.batchNorm, 128, 2),
            predict_flow(self.batchNorm, 128, 2),
            predict_flow(self.batchNorm, 128, 2),
            predict_flow(self.batchNorm, 128, 2)
        ])

    def forward(self, input, image_resize, sp_threshold):
        threshold = sp_threshold
        alpha = self.alpha
        N, _, _, _, T = input.shape

        # Match pyramidal channels: 64, 128, 256, 512
        m1 = torch.zeros(N, 64,  image_resize//2,  image_resize//2).to(input.device)
        m2 = torch.zeros(N, 128, image_resize//4,  image_resize//4).to(input.device)
        m3 = torch.zeros(N, 256, image_resize//8,  image_resize//8).to(input.device)
        m4 = torch.zeros(N, 512, image_resize//16, image_resize//16).to(input.device)

        m1_t, m2_t, m3_t, m4_t = m1.clone(), m2.clone(), m3.clone(), m4.clone()

        for i in range(T):
            frame = input[:, :, :, :, i].to(input.device)

            cur1 = self.conv_s1(frame)
            m1 = alpha * m1 + cur1
            m1, s1 = LIF_Neuron(m1, threshold)
            m1_t += cur1

            cur2 = self.conv_s2(s1)
            m2 = alpha * m2 + cur2
            m2, s2 = LIF_Neuron(m2, threshold)
            m2_t += cur2

            cur3 = self.conv_s3(s2)
            m3 = alpha * m3 + cur3
            m3, s3 = LIF_Neuron(m3, threshold)
            m3_t += cur3

            cur4 = self.conv_s4(s3)
            m4 = alpha * m4 + cur4
            m4, s4 = LIF_Neuron(m4, threshold)
            m4_t += cur4

        blocks = [m1_t, m2_t, m3_t, m4_t] 

        # 2. TPA (Transformer Pyramid) reasoning
        # Scale 0: 16x16 tokens
        t0 = self.split0(blocks[-1]).flatten(2).transpose(1, 2)
        p0 = self.position_embedding(t0)
        hs0 = self.trans_encoder0(src=t0.transpose(0, 1), pos=p0.transpose(0, 1))
        hc0 = self.trans_decoder0(tgt=hs0, memory=hs0)

        # Scale 1: 32x32 tokens
        t1 = self.split1(blocks[-2]).flatten(2).transpose(1, 2)
        p1 = self.position_embedding(t1)
        hs1 = self.trans_encoder1(src=t1.transpose(0, 1), pos=p1.transpose(0, 1))
        hc1 = self.trans_decoder1(tgt=hs1, memory=hs0)

        # Scale 2: 64x64 tokens
        t2 = self.split2(blocks[-3]).flatten(2).transpose(1, 2)
        p2 = self.position_embedding(t2)
        hs2 = checkpoint(self.trans_encoder2, t2.transpose(0, 1), p2.transpose(0, 1), use_reentrant=False)
        hc2 = checkpoint(self.trans_decoder2, hs2, hs1, use_reentrant=False)

        # Scale 3: Cap at 64x64 tokens for efficiency
        t3 = self.split3(blocks[-4]).flatten(2).transpose(1, 2)
        p3 = self.position_embedding(t3)
        hs3 = checkpoint(self.trans_encoder3, t3.transpose(0, 1), p3.transpose(0, 1), use_reentrant=False)
        hc3 = checkpoint(self.trans_decoder3, hs3, hs2, use_reentrant=False)

        H, W = image_resize, image_resize
        hc0_img = rearrange(hc0, '(h w) n c -> n c h w', h=H//16, w=W//16) # 16x16
        hc1_img = rearrange(hc1, '(h w) n c -> n c h w', h=H//8,  w=W//8)  # 32x32
        hc2_img = rearrange(hc2, '(h w) n c -> n c h w', h=H//4,  w=W//4)  # 64x64
        hc3_img = rearrange(hc3, '(h w) n c -> n c h w', h=H//4,  w=W//4)  # 64x64

        # 3. MLU (Multi-Level Upsampler) Hierarchical Fusion
        # Stage 0: Refine to 16x16
        in0 = self.UpsampleConv[0](hc0_img) 
        flow0 = self.predict_flow[0](in0)
        hs_up = self.deconv[0](hc0_img) 

        # Stage 1: Refine to 32x32
        cat1 = torch.cat((flow0, blocks[2], hs_up, hc1_img), 1)
        in1 = self.UpsampleConv[1](cat1) 
        flow1 = self.predict_flow[1](in1)
        cat1_up = self.deconv[1](cat1) 

        # Stage 2: Refine to 64x64
        cat2 = torch.cat((flow1, blocks[1], cat1_up, hc2_img), 1)
        in2 = checkpoint(self.UpsampleConv[2], cat2, use_reentrant=False)
        flow2 = self.predict_flow[2](in2)
        cat2_up = checkpoint(self.deconv[2], cat2, use_reentrant=False)

        # Stage 3: Refine to 128x128
        hc3_up = F.interpolate(hc3_img, scale_factor=2, mode='bilinear', align_corners=False)
        cat3 = torch.cat((flow2, blocks[0], cat2_up, hc3_up), 1)
        in3 = checkpoint(self.UpsampleConv[3], cat3, use_reentrant=False)   
        flow3 = self.predict_flow[3](in3)

        return [flow0, flow1, flow2, flow3]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def let_flownet(args, device, data=None):
    model = LET_FlowNet(args, device)

    if data is not None:
        try:
            model.load_state_dict(data['state_dict'], strict=False)
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")

    return model
