import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beta = 5

class SurrogateGradSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out.type(torch.FloatTensor).to(device)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        if (True in np.isnan(grad_input.cpu().detach())):
            print("NaN detected in grad_input, stop autograd backward")
            return

        grad = grad_input * beta * torch.sigmoid(beta * input) * (1 - torch.sigmoid(beta * input))

        if (True in np.isnan(grad.cpu().detach())):
            print("NaN detected in grad after autograd")
            return

        # print(f"grad[grad < grad_input]: {len(grad[grad < grad_input])}")
        # print(f"grad[grad == grad_input]: {len(grad[grad == grad_input])}")
        # print(f"grad[grad > grad_input]: {len(grad[grad > grad_input])}")

        return grad


def LIF_Neuron(mem, threshold):

    # generate spike
    spike = SurrogateGradSpike.apply(mem - threshold)
    mem = mem * (1 - spike)  # reset after spike

    return mem, spike
