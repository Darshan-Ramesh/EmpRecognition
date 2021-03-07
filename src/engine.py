import torch
import torch.nn as nn
from tqdm import tqdm
import time
import torch.nn.functional as F
import config as cfg
from sklearn.metrics import precision_score, f1_score, recall_score
from torch.cuda import amp


class Logger(object):
    """Logging class to log training progress information such as
    1. Epochs
    2. Training and validation loss
    3. Metric to log (Accuracy, F1 score etc.)
    """

    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x

    def __call__(self, loss, metrics, i):
        track_str = '\r{} | {:5d}/{:<5d}| '.format(
            self.mode, i + 1, self.length)
        loss_str = 'loss: {:9.4f} | '.format(self.fn(loss, i))
        metric_str = ' | '.join('{}: {:9.4f}'.format(
            k, self.fn(v, i)) for k, v in metrics.items())
        print(track_str + loss_str + metric_str + '   ', end='')
        if i + 1 == self.length:
            print('')


class BatchTimer(object):
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.

    Keyword Arguments:
        rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample). (default: {True})
        per_sample {bool} -- Whether to report times or rates per sample or per batch.
            (default: {True})
    """

    def __init__(self, rate=True, per_sample=True):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)


def accuracy(logits, y):
    """Method to calculate acccuracy during training and validation

    Args:
        logits ([Torch tensor]): Output from the last layer of the network, before softmax layer 
        y ([Torch Tensor]): labels of the associated logit

    Returns:
        [Torch Tensor]: Mean accuracy of the batch
    """
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean().detach().cpu()


def f1score(logits, y):
    """Method to calculate F1 score during training and validation

    Args:
        logits ([Torch tensor]): Output from the last layer of the network, before softmax layer 
        y ([Torch Tensor]): labels of the associated logit

    Returns:
        [Torch Tensor]: weighted accuracy of the batch
    """

    _, y_preds = torch.max(logits, 1)
    fscore = f1_score(y_preds.cpu().numpy(),
                      y.cpu().numpy(), average='weighted')
    return fscore


def precision(logits, y):
    """Method to precision during training and validation

    Args:
        logits ([Torch tensor]): Output from the last layer of the network, before softmax layer 
        y ([Torch Tensor]): labels of the associated logit

    Returns:
        [Torch Tensor]: weighted precision of the batch
    """
    _, y_preds = torch.max(logits, 1)
    pscore = precision_score(y_preds.cpu().numpy(),
                             y.cpu().numpy(), average='weighted')
    return pscore


def recall(logits, y):
    """Method to calculate mean recall during training and validation

    Args:
        logits ([Torch tensor]): Output from the last layer of the network, before softmax layer 
        y ([Torch Tensor]): labels of the associated logit

    Returns:
        [Torch Tensor]: weighted recall of the batch
    """
    _, y_preds = torch.max(logits, 1)
    rscore = recall_score(y_preds.cpu().numpy(),
                          y.cpu().numpy(), average='weighted')
    return rscore


def lr(optimizer):
    """Learning rate

    Args:
        optimizer ([dict]): Standard optimizer from torch.optim

    Returns:
        [dict]: parameters related to learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def pass_epoch(model, model_head, loss_fn, data_loader,
               optimizer=None, scheduler=None, batch_metrics={'time': BatchTimer()},
               show_running=None, device='cpu', writer=None, epoch=None,
               grad_scaler=None):
    """Training and updating the optimizers per epoch

    Args:
        model ([torch model]): Standard torch model
        model_head ([torch model]): If there are any heads to the model before sending information to the Dense layers
        loss_fn ([torch loss]): Standard loss function used in torch
        data_loader ([DataLoader]): Standard dataloder which loads the batch of images and its associated labels
        optimizer ([torch optim], optional): Any optimizer constructed using torch modules. Defaults to None.
        scheduler ([torch scheduler], optional): Learning rate schedulers constructed using standard torch modules . Defaults to None.
        batch_metrics (dict, optional): Metrics to track during training. Defaults to {'time': BatchTimer()}.
        show_running ([str], optional): To display the running mean. Defaults to None.
        device (str, optional): device to train on 'cpu' or 'cuda'. Defaults to 'cpu'.
        writer ([object], optional): Logging the training information to Tensorboard. Defaults to None.
        epoch ([int], optional): Current epoch during training. Defaults to None.
        grad_scaler ([object], optional): Related to Automatic mixed precision training. Defaults to None.

    Returns:
        [type]: [description]
    """

    mode = 'Train' if model.training else 'Valid'
    logger = Logger(mode, length=len(data_loader), calculate_mean=show_running)
    loss = 0
    metrics = {}

    for i_batch, (x, y, _) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)

        if model_head:
            # MP training
            if model.training:
                with amp.autocast():
                    embeddings = model(x)
                y_pred, original_logits = model_head(embeddings, y)
            else:
                # validation
                embeddings = model(x)
                y_pred, original_logits = model_head(embeddings, y)

        else:
            if model.training:
                with amp.autocast():
                    y_pred = model(x)
            else:
                y_pred = model(x)

        if grad_scaler and model.training:
            with amp.autocast():
                loss_batch = loss_fn(y_pred, y)
        else:
            loss_batch = loss_fn(y_pred, y)

        if model.training:
            optimizer.zero_grad()
            grad_scaler.scale(loss_batch).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            if model_head:
                metrics_batch[metric_name] = metric_fn(original_logits.data, y)
            else:
                metrics_batch[metric_name] = metric_fn(y_pred, y)

            metrics[metric_name] = metrics.get(
                metric_name, 0) + metrics_batch[metric_name]

        if writer is not None and model.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars(
                    'BatchLoss', {mode: loss_batch.detach().cpu()}, epoch)

                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(
                        metric_name, {mode: metric_batch}, epoch)
            writer.iteration += 1

        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch

        if show_running:
            logger(loss, metrics, i_batch)
        else:
            logger(loss_batch, metrics_batch, i_batch)

    if model.training and scheduler is not None:
        scheduler.step()
        writer.add_scalars('learning rate', {
                           mode: lr(optimizer)}, writer.iteration)

    loss = loss / (i_batch+1)
    metrics = {k: v/(i_batch+1) for k, v in metrics.items()}

    # average training loss + metrics
    if writer is not None and model.training:
        writer.add_scalars('Epoch loss', {mode: loss.detach()}, epoch)

        # writes Accuracy,f1score etc other metrics during training
        # for metric_name, metric in metrics.items():
        #     writer.add_scalars(metric_name,{mode:metric})

    # average validation loss + metrics
    if writer is not None and not model.training:
        writer.add_scalars('Epoch loss', {mode: loss.detach()}, epoch)

        # writes Accuracy,f1score etc other metrics during validation
        # for metric_name, metric in metrics.items():
        #     writer.add_scalars(metric_name, {mode:metric})
    return loss, metrics
