import argparse
import os
import sys
import shutil
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import torchvision.transforms as T
# import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

#import densenet as dn
from model.tiramisu import FCDenseNet103 as dn
import splitdata as split


parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=60, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--sockeye', '-s', type=str, help='Save progress to specified sockeye directory')
parser.add_argument('--job', type = str, help='SLURM job ID')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0
writer = None


def main():
    print(">>> Starting script...")
    global args, best_prec1, writer
    args = parser.parse_args()
    log_dir=f'./runs/{args.name}_{args.job}'
    
    #for logging to tensorboard
    if args.tensorboard:
        writer = SummaryWriter(log_dir=log_dir)

    # Data loading code
    
    kwargs = {'num_workers': 1, 'pin_memory': False}
    
    
    if args.sockeye:
        output_dir = args.sockeye

    else:
        output_dir = "/Users/joannwokeforo/Documents/BMEG457/Data/ISPY1-TestFormat"
    
    #separating dataset into images and masks folders
    image_dir = f"{output_dir}/all_images"
    mask_dir = f"{output_dir}/all_masks"
    
    #split into train, val, and test, datasets
    split.split_data(image_dir,mask_dir,output_dir)

    train_dataset = split.SegDataset(img_dir=f"{output_dir}/train/images", mask_dir=f"{output_dir}/train/masks")
    val_dataset = split.SegDataset(img_dir=f"{output_dir}/val/images", mask_dir=f"{output_dir}/val/masks")

    #takes a long time so creating checkpoint folder to store data
    checkpoints = os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory = True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory = True)

    # create model
    model = dn(1)
    
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
  
    if args.sockeye:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = model.to(device)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if device.type == "cuda":
        cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    print("Loss -> 0.3BCE, 0.7DICE")
    # criterion = dice_loss
    criterion = combined_loss
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             nesterov=True,
    #                             weight_decay=args.weight_decay)

    
    print(">>> Entering epoch loop")
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, device)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, device):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    sens1 = AverageMeter()
    spec1 = AverageMeter()
    acc1 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)
        input_var = input
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var, 0.3, 0.7)

        # measure dice, and record loss
        #prec1 = accuracy(output.data, target, topk=(1,))[0]
        dice = dice_score(output.detach(),target)
        sens = sensitivity(output.detach(), target)
        spec = specificity(output.detach(), target)
        acc = balanced_accuracy(output.detach(), target)
        
        losses.update(loss.item(), input.size(0))
        top1.update(dice.item(), input.size(0))
        sens1.update(sens.item(), input.size(0))
        spec1.update(spec.item(), input.size(0))
        acc1.update(acc.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Dice {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
    #optional logging
    if writer is not None:
        # log to TensorBoard
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/dce', top1.avg, epoch)
        writer.add_scalar('train/sensitivity', sens1.avg, epoch)
        writer.add_scalar('train/specificity', spec1.avg, epoch)
        writer.add_scalar('train/accuracy', acc1.avg, epoch)    

def validate(val_loader, model, criterion, epoch, device):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    sens1 = AverageMeter()
    spec1 = AverageMeter()
    acc1 = AverageMeter()


    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target, 0.3, 0.7)

        if i == 0 and epoch % 5 == 0:
            step = epoch
            fg_idxs = (target.sum(dim=(1,2,3)) > 0).nonzero(as_tuple=True)[0]
            if len(fg_idxs) > 0:
                log_img_tensorboard(writer, input, target, output, step, idx=fg_idxs[0].item())


        
        # measure accuracy and record loss
        dice = dice_score(output.detach(),target)
        sens = sensitivity(output.detach(), target)
        spec = specificity(output.detach(), target)
        acc = balanced_accuracy(output.detach(), target)
        
        losses.update(loss.item(), input.size(0))
        top1.update(dice.item(), input.size(0))
        sens1.update(sens.item(), input.size(0))
        spec1.update(spec.item(), input.size(0))
        acc1.update(acc.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Dice {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Dice {top1.avg:.3f}\t'
          'Sensitivity {sens.avg:.3f}\t'
          'Specificity {spec.avg:.3f}\t'
          'Accuracy {acc.avg:.3f}'.format(top1=top1, sens=sens1, spec=spec1, acc=acc1))
    
    
    if writer is not None:
        # log to TensorBoard
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/dce', top1.avg, epoch)
        writer.add_scalar('val/sensitivity', sens1.avg, epoch)
        writer.add_scalar('val/specificity', spec1.avg, epoch)
        writer.add_scalar('val/accuracy', acc1.avg, epoch)
    
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # log to TensorBoard
    if writer is not None:
        writer.add_scalar('learning_rate', lr, epoch)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def sensitivity (output, target, smooth = 1e-6):
    pred = torch.sigmoid(output)
    pred_bin = (pred > 0.5).float()
    target = target.float()

    true_positive = (pred_bin * target).sum()
    false_negative = ((1 - pred_bin) * target).sum()

    return (true_positive + smooth) / (true_positive + false_negative + smooth)

def specificity (output, target, smooth = 1e-6):
    pred = torch.sigmoid(output)
    pred_bin = (pred > 0.5).float()
    target = target.float()

    true_negative = ((1-pred_bin) * (1 - target)).sum()

    false_positive = (pred_bin * (1 - target)).sum()

    return (true_negative + smooth) / (true_negative + false_positive + smooth)

def balanced_accuracy(output, target):
    return (sensitivity(output,target) + specificity(output,target)) / 2

def dice_score(output, target, smooth = 1e-6):
    """Computes the dice score for the model output and ground truth"""
    
    pred = torch.sigmoid(output)
    target = target.float()
    
    intersection = (pred * target).sum(dim=(1, 2, 3)) 
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def dice_loss(pred, target, smooth = 1):
    """Computes the dice loss for the model prediction and ground truth"""
    dce = dice_score(pred,target)
    return 1 - dce

def bce_loss(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target.float())

def combined_loss(pred, target, bce_weight=0.5, dce_weight=0.5):
    if target.dim() == 3:
        target = target.unsqueeze(1)
    target = target.float()
    
    bce = nn.BCEWithLogitsLoss()(pred, target.float())
    d_loss = dice_loss(pred, target.float())

    return bce_weight * bce + dce_weight * d_loss

def log_img_tensorboard(writer, input, target, output, step, idx=0):
    """
    input:  [B, 1, H, W]
    target: [B, 1, H, W]
    output: [B, 1, H, W] (logits)
    """
    img = input[idx:idx+1]
    gt  = target[idx:idx+1].float()
    pred = (torch.sigmoid(output[idx:idx+1]) > 0.5).float()

    grid = torch.cat([img, gt, pred], dim=0)

    writer.add_images(tag= "val/sample", img_tensor=grid, global_step = step)


if __name__ == '__main__':
    main()
