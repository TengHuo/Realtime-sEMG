# -*- coding: UTF-8 -*-
# torchsummary.py
# modified from https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
# thanks sksq96
#

import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size, device):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                if isinstance(module, torch.nn.RNNBase):
                    summary[m_key]["output_shape"] = list(output[0].size())
                    summary[m_key]["output_shape"][0] = batch_size
                else:
                    summary[m_key]["output_shape"] = [
                        [batch_size] + list(o.size())[1:] for o in output
                    ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if isinstance(module, nn.RNNBase):
                summary[m_key]["trainable"] = module.all_weights[0][0].requires_grad
                for weight in module.all_weights[0]:
                    params += torch.prod(torch.LongTensor(list(weight.size())))
            else:
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_content = "----------------------------------------------------------------\n"
    summary_content += "{:>20}  {:>25} {:>15}\n".format("Layer (type)", "Output Shape", "Param #")
    summary_content += "================================================================\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}\n".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_content += line_new

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_content += "================================================================\n"
    summary_content += "Total params: {0:,}\n".format(total_params)
    summary_content += "Trainable params: {0:,}\n".format(trainable_params)
    summary_content += "Non-trainable params: {0:,}\n".format(total_params - trainable_params)
    summary_content += "----------------------------------------------------------------\n"
    summary_content += "Input size (MB): %0.2f\n" % total_input_size
    summary_content += "Forward/backward pass size (MB): %0.2f\n" % total_output_size
    summary_content += "Params size (MB): %0.2f\n" % total_params_size
    summary_content += "Estimated Total Size (MB): %0.2f\n" % total_size
    summary_content += "----------------------------------------------------------------"
    # return summary
    return summary_content
