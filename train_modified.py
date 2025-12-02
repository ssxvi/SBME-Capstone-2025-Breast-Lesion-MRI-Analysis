import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

#import densenet as dn
from model.tiramisu import FCDenseNet103 as dn
import splitdata as split

# used for logging to TensorBoard
# create a writer object

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
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

best_prec1 = 0

def main():
    # print(">>> Starting script...")
    # print("Using device:", torch.device("cpu"))
    global args, best_prec1, writer
    args = parser.parse_args()
    log_dir=f'./runs/{args.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    #for logging to tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # Data loading code
    
    kwargs = {'num_workers': 1, 'pin_memory': False}
    
    #separating images into images and masks folders
    output_dir = "/Users/joannwokeforo/Documents/BMEG457/Data/ISPY1-TestFormat"
    image_dir = f"{output_dir}/all_images"
    mask_dir = f"{output_dir}/all_masks"
    
    #split into train, val, and test, datasets
    split.split_data(image_dir,mask_dir,output_dir)

    train_dataset = split.SegDataset(img_dir=f"{output_dir}/train/images", mask_dir=f"{output_dir}/train/masks")
    val_dataset = split.SegDataset(img_dir=f"{output_dir}/val/images", mask_dir=f"{output_dir}/val/masks")
    #takes a long time so creating checkpoint folder to store data
    checkpoints = os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    #End Joan Edit
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False, num_workers=4)

    # create model
    model = dn(1)
    
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
  
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

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = combined_loss
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    
    # print(">>> Starting training...")
    # print("Train batches:", len(train_loader))
    # print("Val batches:", len(val_loader))
    # print("Model device:", next(model.parameters()).device)
    for epoch in range(args.start_epoch, args.epochs):
        # for i, (input, target) in enumerate(train_loader):
        #     if i == 0:
        #         print(f"Entered training loop. First batch shape: input={input.shape}, target={target.shape}")
        #     if i % args.print_freq == 0:
        #         print(f"[Epoch {epoch} | Batch {i}] Still running...")


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
        # print(f">>> Finished epoch {epoch}")
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

        # if i == 0:
        #   print("TRAIN BATCH DEBUG")
        #   print("input shape:", input.shape)
        #   print("target unique:", torch.unique(target))

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

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
            # print("input min/max:", input.min().item(), input.max().item())
            # print("target min/max:", target.min().item(), target.max().item())
            # print("output min/max:", output.min().item(), output.max().item())
            loss = criterion(output, target)

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

    return (true_positive + smooth) / (target.sum() + smooth)

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
    # print("output min/max", output.min().item(), output.max().item())
    
    pred = torch.sigmoid(output)
    
    pred_bin = (pred > 0.5).float()
    target = target.float()

    # print("pred min/max:", pred.min().item(), pred.max().item())
    # print("pred_bin min/max:", pred_bin.min().item(), pred_bin.max().item())
    # print("target min/max:", target.min().item(), target.max().item())
    
    intersection = (pred_bin * target).sum() 
    union = pred_bin.sum() + target.sum()
    # print("Intersection:", intersection)
    # print("Union", union)

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(pred, target, smooth = 1):
    """Computes the dice loss for the model prediction and ground truth"""
    return 1 - dice_score(pred, target)

def combined_loss(pred, target):
    if target.dim() == 3:
        target = target.unsqueeze(1)
    target = target.float()
    
    bce = nn.BCEWithLogitsLoss()(pred, target.float())
    d_loss = dice_loss(pred, target.float())

    return 0.5 * bce + 0.5 * d_loss


if __name__ == '__main__':
    main()
