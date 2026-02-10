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


__all__ = ['spiket_flownet_snn_lif_trans']


class SpikeT_FlowNet_SNN_LIF_Trans(BaseModel):
    expansion = 1

    def __init__(self, args, device, batchNorm=True):
        super(SpikeT_FlowNet_SNN_LIF_Trans, self).__init__()

        self.args = args
        self.device = device

        # SNN
        self.batchNorm = batchNorm
        self.conv_s1 = conv_s(self.batchNorm, 4, 64, stride=2)
        self.conv_s2 = conv_s(self.batchNorm, 64, 128, stride=2)
        self.conv_s3 = conv_s(self.batchNorm, 128, 256, stride=2)
        self.conv_s4 = conv_s(self.batchNorm, 256, 512, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

        self.dt = 10 * 1e-3
        self.alpha = np.exp(-self.dt/self.args.tau)

        # Transformers
        norm = self.args.norm
        self.head = ConvLayer(in_channels=4, out_channels=64, kernel_size=5, stride=1, padding=2, norm=norm)

        self.position_embedding = build_position_encoding('sine', 512)
        self.split0 = nn.Unfold(kernel_size=1, stride=1, padding=0)

        num_enc_layers = self.args.num_enc_layers
        num_dec_layers = self.args.num_dec_layers

        # TODO: 1. try different dropout
        self.trans_encoder0 = transformer_encoder(d_model=512, nhead=8, num_encoder_layers=num_enc_layers,
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        self.trans_decoder0 = transformer_decoder(d_model=512, nhead=8, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        
        self.split1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=0)
        self.trans_encoder1 = transformer_encoder(d_model=512, nhead=8, num_encoder_layers=num_enc_layers, 
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        self.trans_decoder1 = transformer_decoder(d_model=512, nhead=8, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        
        self.split2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=4, stride=4, padding=0)
        self.trans_encoder2 = transformer_encoder(d_model=512, nhead=8, num_encoder_layers=num_enc_layers, 
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        self.trans_decoder2 = transformer_decoder(d_model=512, nhead=8, num_decoder_layers=num_dec_layers,
                                                dim_feedforward=1024, activation='relu', dropout=0.1)
        
        self.split3 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=8, stride=8, padding=0)
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
            predict_flow(self.batchNorm, 128, 2)
        ])


    def forward(self, input, image_resize, sp_threshold):

        # Encoder-SNN: temporal feature extraction
        threshold = sp_threshold
        alpha = self.alpha

        mem_1 = torch.zeros(input.size(0), 64, int(
            image_resize/2), int(image_resize/2)).to(input.device)
        mem_2 = torch.zeros(input.size(0), 128, int(
            image_resize/4), int(image_resize/4)).to(input.device)
        mem_3 = torch.zeros(input.size(0), 256, int(
            image_resize/8), int(image_resize/8)).to(input.device)
        mem_4 = torch.zeros(input.size(0), 512, int(
            image_resize/16), int(image_resize/16)).to(input.device)

        mem_1_total = torch.zeros(input.size(0), 64, int(
            image_resize/2), int(image_resize/2)).to(input.device)
        mem_2_total = torch.zeros(input.size(0), 128, int(
            image_resize/4), int(image_resize/4)).to(input.device)
        mem_3_total = torch.zeros(input.size(0), 256, int(
            image_resize/8), int(image_resize/8)).to(input.device)
        mem_4_total = torch.zeros(input.size(0), 512, int(
            image_resize/16), int(image_resize/16)).to(input.device)

        for i in range(input.size(4)):
            input11 = input[:, :, :, :, i].to(input.device)

            current_1 = self.conv_s1(input11)
            mem_1 = alpha*mem_1 + current_1
            mem_1, spike_1 = LIF_Neuron(mem_1, threshold)
            mem_1_total = mem_1_total + current_1

            current_2 = self.conv_s2(spike_1)
            mem_2 = alpha*mem_2 + current_2
            mem_2, spike_2 = LIF_Neuron(mem_2, threshold)
            mem_2_total = mem_2_total + current_2

            current_3 = self.conv_s3(spike_2)
            mem_3 = alpha*mem_3 + current_3
            mem_3, spike_3 = LIF_Neuron(mem_3, threshold)
            mem_3_total = mem_3_total + current_3

            current_4 = self.conv_s4(spike_3)
            mem_4 = alpha*mem_4 + current_4
            mem_4, spike_4 = LIF_Neuron(mem_4, threshold)
            mem_4_total = mem_4_total + current_4

        blocks = []
        # Big -> Small
        blocks.append(mem_1_total)
        blocks.append(mem_2_total)
        blocks.append(mem_3_total)
        blocks.append(mem_4_total)

        # Encoder-Transformers: Token Pyramid Aggregation (TPA) for global spatial context extraction
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        #************* path to transformer
        # TODO: 5 - verify the impact of using the event at the first time slot
        head = self.head(input[:, :, :, :, 0])
        
        n, c, H, W = head.size()

        # Small -> Big
        #******** scale 0
        token0 = self.split0(blocks[-1]).transpose(1, 2)
        pos0 = self.position_embedding(token0)
        hs0 = self.trans_encoder0(src=token0.transpose(0, 1), pos=pos0.transpose(0, 1))
        hc0 = self.trans_decoder0(tgt=hs0, memory=hs0)

        #******** scale 1
        token1 = self.split1(blocks[-2]).flatten(2).transpose(1, 2)
        pos1 = self.position_embedding(token1)
        hs1 = self.trans_encoder1(src=token1.transpose(0, 1), pos=pos1.transpose(0, 1))
        hc1 = self.trans_decoder1(tgt=hs1, memory=hs0)

        #******** scale 2
        token2 = self.split2(blocks[-3]).flatten(2).transpose(1, 2)
        pos2 = self.position_embedding(token2)
        hs2 = self.trans_encoder2(src=token2.transpose(0, 1), pos=pos2.transpose(0, 1))
        hc2 = self.trans_decoder2(tgt=hs2, memory=hs1)

        #******** scale 3
        # token3 = self.split3(head).flatten(2).transpose(1, 2)
        token3 = self.split3(blocks[-4]).flatten(2).transpose(1, 2)
        pos3 = self.position_embedding(token3)
        hs3 = self.trans_encoder3(src=token3.transpose(0, 1), pos=pos3.transpose(0, 1))
        hc3 = self.trans_decoder3(tgt=hs3, memory=hs2)

        # Hierarchical decoding with skip connections from transformer decoders
        # to address the feature aggregation bottleneck.
        hc0_img = rearrange(hc0, '(h w) n c -> n c h w', h=H//16, w=W//16)
        hc1_img = rearrange(hc1, '(h w) n c -> n c h w', h=H//16, w=W//16)
        hc2_img = rearrange(hc2, '(h w) n c -> n c h w', h=H//16, w=W//16)
        hc3_img = rearrange(hc3, '(h w) n c -> n c h w', h=H//16, w=W//16)

        # Decoder & Prediction: Multi-Level Upsampler (MLU)

        # Small -> Big
        # Start with the smallest scale transformer output
        input0 = self.UpsampleConv[0](hc0_img)
        flow0 = self.predict_flow[0](input0)
        hs_up = self.deconv[0](hc0_img)

        # Upsample and inject next scale transformer output
        hc1_up = F.interpolate(hc1_img, scale_factor=2, mode='bilinear', align_corners=False)
        concat1 = torch.cat((flow0, blocks[2], hs_up, hc1_up), 1)
        input1 = self.UpsampleConv[1](concat1)
        flow1 = self.predict_flow[1](input1)
        concat1_up = self.deconv[1](concat1)

        # Upsample and inject next scale transformer output
        hc2_up = F.interpolate(hc2_img, scale_factor=4, mode='bilinear', align_corners=False)
        concat2 = torch.cat((flow1, blocks[1], concat1_up, hc2_up), 1)
        input2 = self.UpsampleConv[2](concat2)
        flow2 = self.predict_flow[2](input2)
        concat2_up = self.deconv[2](concat2)

        # Upsample and inject final scale transformer output
        hc3_up = F.interpolate(hc3_img, scale_factor=8, mode='bilinear', align_corners=False)
        concat3 = torch.cat((flow2, blocks[0], concat2_up, hc3_up), 1)
        input3 = self.UpsampleConv[3](concat3)
        flow3 = self.predict_flow[3](input3)

        return flow3

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def spiket_flownet_snn_lif_trans(args, device, data=None):
    model = SpikeT_FlowNet_SNN_LIF_Trans(args, device)

    if data is not None:
        model.load_state_dict(data['state_dict'])

    return model
