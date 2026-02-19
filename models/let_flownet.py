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

        # 1. SNN Encoder: Unified to 128 channels across all scales
        self.batchNorm = batchNorm
        self.conv_s1 = conv_s(self.batchNorm, 4, 128, stride=2)
        self.conv_s2 = conv_s(self.batchNorm, 128, 128, stride=2)
        self.conv_s3 = conv_s(self.batchNorm, 128, 128, stride=2)
        self.conv_s4 = conv_s(self.batchNorm, 128, 128, stride=2)

        # use 3 for dt1, and 2 for dt4 and larger (e.g. dt8)
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

        # 2. TPA (Transformer Pyramid): Unified d_model = 128
        self.head = ConvLayer(in_channels=4, out_channels=64, kernel_size=5, stride=1, padding=2, norm=norm)
        
        # Initialize with n_position=16384 to support 128x128 tokens at scale 3
        self.position_embedding = build_position_encoding('sine', 128, n_position=16384)
        
        self.split0 = nn.Unfold(kernel_size=1, stride=1, padding=0)

        num_enc_layers = self.args.num_enc_layers
        num_dec_layers = self.args.num_dec_layers
        d_model = 128
        dim_ff = 256 # Reduced FFN to save memory

        # Chained Scale-Aware Transformers
        self.trans_encoder0 = transformer_encoder(d_model=d_model, nhead=4, num_encoder_layers=num_enc_layers,
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        self.trans_decoder0 = transformer_decoder(d_model=d_model, nhead=4, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        
        # Scale 1-3 Tokenizers: 128 -> 128 with stride=1 for native resolution
        self.split1 = ConvLayer(128, d_model, kernel_size=3, stride=1, padding=1, norm=norm)
        self.trans_encoder1 = transformer_encoder(d_model=d_model, nhead=4, num_encoder_layers=num_enc_layers, 
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        self.trans_decoder1 = transformer_decoder(d_model=d_model, nhead=4, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        
        self.split2 = ConvLayer(128, d_model, kernel_size=3, stride=1, padding=1, norm=norm)
        self.trans_encoder2 = transformer_encoder(d_model=d_model, nhead=4, num_encoder_layers=num_enc_layers, 
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        self.trans_decoder2 = transformer_decoder(d_model=d_model, nhead=4, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        
        self.split3 = ConvLayer(128, d_model, kernel_size=3, stride=1, padding=1, norm=norm)
        self.trans_encoder3 = transformer_encoder(d_model=d_model, nhead=4, num_encoder_layers=num_enc_layers, 
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)
        self.trans_decoder3 = transformer_decoder(d_model=d_model, nhead=4, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=dim_ff, activation='relu', dropout=0.1)

        # 3. MLU (Multi-Level Upsampler): Hierarchical concatenation
        # concat_in = flow(2) + block(128) + deconv(128) + transformer_skip(128) = 386
        self.deconv = nn.ModuleList([
            deconv(self.batchNorm, 128, 128),
            deconv(self.batchNorm, 386, 128),
            deconv(self.batchNorm, 386, 128)
        ])
        
        self.UpsampleConv = nn.ModuleList([
            UpsampleConvLayer(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm),
            UpsampleConvLayer(in_channels=386, out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm),
            UpsampleConvLayer(in_channels=386, out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm),
            UpsampleConvLayer(in_channels=386, out_channels=128, kernel_size=5, stride=1, padding=2, norm=norm)
        ])

        self.predict_flow = nn.ModuleList([
            predict_flow(self.batchNorm, 128, 2),
            predict_flow(self.batchNorm, 128, 2),
            predict_flow(self.batchNorm, 128, 2),
            predict_flow(self.batchNorm, 128, 2)
        ])

    def forward(self, input, image_resize, sp_threshold):
        # 1. Encoder-SNN: Unified 128-channel temporal feature extraction
        #
        threshold = sp_threshold
        alpha = self.alpha
        N, _, _, _, T = input.shape

        # Initialize memories at 128 channels for all scales
        #
        m1 = torch.zeros(N, 128, image_resize//2,  image_resize//2).to(input.device)
        m2 = torch.zeros(N, 128, image_resize//4,  image_resize//4).to(input.device)
        m3 = torch.zeros(N, 128, image_resize//8,  image_resize//8).to(input.device)
        m4 = torch.zeros(N, 128, image_resize//16, image_resize//16).to(input.device)

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

        blocks = [m1_t, m2_t, m3_t, m4_t] # [128, 64, 32, 16] resolutions

        # 2. TPA (Transformer Pyramid): Hierarchical Tokenization
        #
        
        # Scale 0: 16x16 tokens (mem_4)
        t0 = self.split0(blocks[-1]).transpose(1, 2)
        p0 = self.position_embedding(t0)
        hs0 = self.trans_encoder0(src=t0.transpose(0, 1), pos=p0.transpose(0, 1))
        hc0 = self.trans_decoder0(tgt=hs0, memory=hs0)

        # Scale 1: 32x32 tokens (mem_3)
        t1 = self.split1(blocks[-2]).flatten(2).transpose(1, 2)
        p1 = self.position_embedding(t1)
        hs1 = self.trans_encoder1(src=t1.transpose(0, 1), pos=p1.transpose(0, 1))
        hc1 = self.trans_decoder1(tgt=hs1, memory=hs0) # Chained memory

        # Scale 2: 64x64 tokens (mem_2) - Implement Checkpointing
        t2 = self.split2(blocks[-3]).flatten(2).transpose(1, 2)
        p2 = self.position_embedding(t2)

        # Wrap encoder and decoder separately to maximize memory savings
        hs2 = checkpoint(self.trans_encoder2, t2.transpose(0, 1), p2.transpose(0, 1), use_reentrant=False)
        hc2 = checkpoint(self.trans_decoder2, hs2, hs1, use_reentrant=False)

        # Scale 3: 128x128 tokens (mem_1) - Implement Checkpointing
        t3 = self.split3(blocks[-4]).flatten(2).transpose(1, 2)
        p3 = self.position_embedding(t3)

        hs3 = checkpoint(self.trans_encoder3, t3.transpose(0, 1), p3.transpose(0, 1), use_reentrant=False)
        hc3 = checkpoint(self.trans_decoder3, hs3, hs2, use_reentrant=False)

        # Reshape tokens back to spatial maps at native resolutions
        #
        H, W = image_resize, image_resize
        hc0_img = rearrange(hc0, '(h w) n c -> n c h w', h=H//16, w=W//16) # 16x16
        hc1_img = rearrange(hc1, '(h w) n c -> n c h w', h=H//8,  w=W//8)  # 32x32
        hc2_img = rearrange(hc2, '(h w) n c -> n c h w', h=H//4,  w=W//4)  # 64x64
        hc3_img = rearrange(hc3, '(h w) n c -> n c h w', h=H//2,  w=W//2)  # 128x128

        # 3. MLU (Multi-Level Upsampler): Refined Hierarchical Concatenation
        #

        # Stage 0: Smallest scale (16x16)
        in0 = self.UpsampleConv[0](hc0_img) # Upsamples to 32x32
        flow0 = self.predict_flow[0](in0)
        hs_up = self.deconv[0](hc0_img) # Upsamples to 32x32

        # Stage 1: Refine to 32x32
        # Concat: flow(2) + block[2](128) + deconv(128) + transformer(128) = 386
        cat1 = torch.cat((flow0, blocks[2], hs_up, hc1_img), 1)
        in1 = self.UpsampleConv[1](cat1) # Upsamples to 64x64
        flow1 = self.predict_flow[1](in1)
        cat1_up = self.deconv[1](cat1) # Upsamples to 64x64

        # Stage 2: Refine to 64x64 - Checkpointing the upsampling fusion
        cat2 = torch.cat((flow1, blocks[1], cat1_up, hc2_img), 1)
        in2 = checkpoint(self.UpsampleConv[2], cat2, use_reentrant=False)
        flow2 = self.predict_flow[2](in2)
        cat2_up = checkpoint(self.deconv[2], cat2, use_reentrant=False)

        # Stage 3: Refine to 128x128 - Critical for OOM
        cat3 = torch.cat((flow2, blocks[0], cat2_up, hc3_img), 1)
        in3 = checkpoint(self.UpsampleConv[3], cat3, use_reentrant=False)   # Upsamples to 256x256
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
            print("Continuing without pre-trained weights.")

    return model
