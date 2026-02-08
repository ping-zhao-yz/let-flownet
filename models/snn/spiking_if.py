import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpikingNN(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return input.gt(1e-5).type(torch.FloatTensor).to(device)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        if (True in np.isnan(grad_input.cpu().detach())):
            print("NaN detected in grad_input, stop autograd backward")
            return

        grad_input[input <= 1e-5] = 0
        return grad_input


def IF_Neuron(membrane_potential, threshold):
    global threshold_k
    threshold_k = threshold

    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold_k, 0)

    # generate spike
    out = SpikingNN.apply(ex_membrane)
    out = out.detach() + (1/threshold)*out - (1/threshold)*out.detach()

    membrane_potential = membrane_potential - ex_membrane  # hard reset

    return membrane_potential, out
