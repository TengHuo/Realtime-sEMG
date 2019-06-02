# -*- coding: UTF-8 -*-
# progressbar.py
# @Time     : 01/Jun/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


import numpy as np
import tqdm

from skorch.dataset import get_len
from skorch.callbacks import Callback


class ProgressBar(Callback):
    """Display a progress bar for each epoch.

    The progress bar includes elapsed and estimated remaining time for
    the current epoch, the number of batches processed, and other
    user-defined metrics. The progress bar is erased once the epoch is
    completed.

    ``ProgressBar`` needs to know the total number of batches per
    epoch in order to display a meaningful progress bar. By default,
    this number is determined automatically using the dataset length
    and the batch size. If this heuristic does not work for some
    reason, you may either specify the number of batches explicitly
    or let the ``ProgressBar`` count the actual number of batches in
    the previous epoch.

    For jupyter notebooks a non-ASCII progress bar can be printed
    instead. To use this feature, you need to have `ipywidgets
    <https://ipywidgets.readthedocs.io/en/stable/user_install.html>`_
    installed.

    Parameters
    ----------

    batches_per_epoch : int, str (default='auto')
      Either a concrete number or a string specifying the method used
      to determine the number of batches per epoch automatically.
      ``'auto'`` means that the number is computed from the length of
      the dataset and the batch size. ``'count'`` means that the
      number is determined by counting the batches in the previous
      epoch. Note that this will leave you without a progress bar at
      the first epoch.

    detect_notebook : bool (default=True)
      If enabled, the progress bar determines if its current environment
      is a jupyter notebook and switches to a non-ASCII progress bar.

    postfix_keys : list of str (default=['train_loss', 'valid_loss'])
      You can use this list to specify additional info displayed in the
      progress bar such as metrics and losses. A prerequisite to this is
      that these values are residing in the history on batch level already,
      i.e. they must be accessible via

      # >>> net.history[-1, 'batches', -1, key]
    """

    def __init__(self, batches_per_epoch='auto', detect_notebook=True, postfix_keys=None):
        self.batches_per_epoch = batches_per_epoch
        self.detect_notebook = detect_notebook
        self.postfix_keys = postfix_keys or ['train_loss']
        self.pbar = None
        self.epoch_step = 0

    def _get_postfix_dict(self, net, epoch_step):
        postfix = {'Epoch': epoch_step}
        for key in self.postfix_keys:
            try:
                postfix[key] = '{:^7.3f}'.format(net.history[-1, 'batches', -1, key])
            except KeyError:
                pass
        return postfix

    def on_train_begin(self, net,
                       X=None, y=None, **kwargs):
        """Called at the beginning of training."""
        self.epoch_step = 0

    # pylint: disable=attribute-defined-outside-init
    def on_batch_end(self, net, **kwargs):
        self.pbar.set_postfix(self._get_postfix_dict(net, self.epoch_step), refresh=False)
        self.pbar.update()

    # pylint: disable=attribute-defined-outside-init, arguments-differ
    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        # Assume it is a number until proven otherwise.
        self.epoch_step += 1
        batches_per_epoch = self.batches_per_epoch

        if self.batches_per_epoch == 'auto':
            batches_per_epoch = _get_batches_per_epoch(
                net, dataset_train, dataset_valid
            )
        elif self.batches_per_epoch == 'count':
            if len(net.history) <= 1:
                # No limit is known until the end of the first epoch.
                batches_per_epoch = None
            else:
                batches_per_epoch = len(net.history[-2, 'batches'])

        self.pbar = tqdm.tqdm(total=batches_per_epoch, leave=False)

    def on_epoch_end(self, net, **kwargs):
        self.pbar.close()


def _get_batches_per_epoch(net, dataset_train, dataset_valid):
    return (_get_batches_per_epoch_phase(net, dataset_train, True) +
            _get_batches_per_epoch_phase(net, dataset_valid, False))


def _get_batches_per_epoch_phase(net, dataset, training):
    if dataset is None:
        return 0
    batch_size = _get_batch_size(net, training)
    return int(np.ceil(get_len(dataset) / batch_size))


def _get_batch_size(net, training):
    name = 'iterator_train' if training else 'iterator_valid'
    net_params = net.get_params()
    return net_params.get(name + '__batch_size', net_params['batch_size'])
