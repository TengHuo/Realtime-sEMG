from .report_logger import ReportLog
from .progressbar import ProgressBar
from .tensorboard_logger import TensorboardCallback
from .torchsummary import summary

from torch import nn
import os

__all__ = [
    'ProgressBar',
    'ReportLog',
    'TensorboardCallback',
    'summary',
    'init_parameters',
    'generate_folder']


def init_parameters(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.RNNBase):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


def generate_folder(root_folder: str, folder_name: str, sub_folder=''):
    # create a folder for storing the model
    root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
    folder = os.path.join(root_path, root_folder, folder_name)
    if sub_folder:
        folder = os.path.join(folder, sub_folder)
    # create a folder for this model
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder
