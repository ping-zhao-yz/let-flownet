import torch
from torch.autograd import Function
from .python import Forward_Warp_Python

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    try:
        import forward_warp_cuda
    except ImportError:
        forward_warp_cuda = None
else:
    forward_warp_cuda = None
    

class forward_warp_fn(Function):

    @staticmethod
    # TODO, suspect "Bilinear" causes model weights not updating after certain threshold during back-propagation.
    def forward(ctx, im0, flow, interpolation_mode_txt = "Bilinear"):

        # permute flow and persist to ctx
        flow = flow.permute(0, 2, 3, 1)

        assert(interpolation_mode_txt in ("Bilinear", "Nearest"))
        if(interpolation_mode_txt == "Bilinear"):
            interpolation_mode = 0
        else:
            interpolation_mode = 1

        '''
        im0: the first image with shape [B, C, H, W]
        flow: the optical flow with shape [B, H, W, 2] (different to grid_sample, it's range is from [-W, -H] to [W, H])
        '''
        assert(len(im0.shape) == len(flow.shape) == 4)
        assert(im0.shape[0] == flow.shape[0])
        assert(im0.shape[-2:] == flow.shape[1:3])
        assert(flow.shape[3] == 2)

        ctx.interpolation_mode = interpolation_mode
        ctx.save_for_backward(im0, flow)

        if im0.is_cuda and forward_warp_cuda is not None:
            im1 = forward_warp_cuda.forward(im0, flow, interpolation_mode)
        else:
            im1 = Forward_Warp_Python.forward(im0, flow, interpolation_mode)

        return im1

    @staticmethod
    def backward(ctx, grad_output):
        im0, flow = ctx.saved_tensors
        if grad_output.is_cuda and forward_warp_cuda is not None:
            im0_grad, flow_grad = forward_warp_cuda.backward(
                grad_output, im0, flow, ctx.interpolation_mode)
        else:
            im0_grad, flow_grad = Forward_Warp_Python.backward(
                grad_output, im0, flow, ctx.interpolation_mode)
        
        # revert - # permute flow and persist to ctx
        flow_grad = flow_grad.permute(0, 3, 1, 2)

        return im0_grad, flow_grad, None
