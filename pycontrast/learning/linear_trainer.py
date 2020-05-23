from __future__ import print_function

import os
import sys
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict

from .util import AverageMeter, accuracy
from .base_trainer import BaseTrainer


class LinearTrainer(BaseTrainer):
    """trainer for Linear evaluation"""
    def __init__(self, args):
        super(LinearTrainer, self).__init__(args)

    def logging(self, epoch, logs, lr=None, train=True):
        """ logging to tensorboard

        Args:
          epoch: training epoch
          logs: loss and accuracy
          lr: learning rate
          train: True of False
        """
        args = self.args
        if args.rank == 0:
            pre = 'train_' if train else 'test_'
            self.logger.log_value(pre+'acc', logs[0], epoch)
            self.logger.log_value(pre+'acc5', logs[1], epoch)
            self.logger.log_value(pre+'loss', logs[2], epoch)
            if train and (lr is not None):
                self.logger.log_value('learning_rate', lr, epoch)

    def wrap_up(self, model, classifier):
        """Wrap up models with DDP

        Args:
          model: pretrained encoder, should be frozen
          classifier: linear classifier
        """
        args = self.args
        model = model.cuda()
        classifier = classifier.cuda()
        model.eval()
        model = DDP(model, device_ids=[args.gpu])
        classifier = DDP(classifier, device_ids=[args.gpu])

        return model, classifier

    def load_encoder_weights(self, model):
        """load pre-trained weights for encoder

        Args:
          model: pretrained encoder, should be frozen
        """
        args = self.args
        if args.ckpt:
            ckpt = torch.load(args.ckpt, map_location='cpu')
            state_dict = ckpt['model']
            if args.modal == 'RGB':
                # Unimodal (RGB) case
                encoder_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('module.', '')
                    if 'encoder' in k:
                        k = k.replace('encoder.', '')
                        encoder_state_dict[k] = v
                model.encoder.load_state_dict(encoder_state_dict)
            else:
                # Multimodal (CMC) case
                encoder1_state_dict = OrderedDict()
                encoder2_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    k = k.replace('module.', '')
                    if 'encoder1' in k:
                        k = k.replace('encoder1.', '')
                        encoder1_state_dict[k] = v
                    if 'encoder2' in k:
                        k = k.replace('encoder2.', '')
                        encoder2_state_dict[k] = v
                model.encoder1.load_state_dict(encoder1_state_dict)
                model.encoder2.load_state_dict(encoder2_state_dict)
            print('Pre-trained weights loaded!')
        else:
            print('==============================')
            print('warning: no pre-trained model!')
            print('==============================')

        return model

    def resume_model(self, classifier, optimizer):
        """load classifier checkpoint"""
        args = self.args
        start_epoch = 1
        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                args.start_epoch = checkpoint['epoch'] + 1
                classifier.load_state_dict(checkpoint['classifier'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                del checkpoint
                torch.cuda.empty_cache()
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        return start_epoch

    def save(self, classifier, optimizer, epoch):
        """save classifier to checkpoint"""
        args = self.args
        if args.local_rank == 0:
            # saving the classifier to each instance
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(args.model_folder, 'current.pth')
            torch.save(state, save_file)
            if epoch % args.save_freq == 0:
                save_file = os.path.join(
                    args.model_folder, 'ckpt_epoch_{}.pth'.format(epoch))
                torch.save(state, save_file)
                # help release GPU memory
            del state

    def train(self, epoch, train_loader, model, classifier,
              criterion, optimizer):
        time1 = time.time()
        args = self.args

        model.eval()
        classifier.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for idx, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            input = input.float()
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # forward
            with torch.no_grad():
                feat = model(x=input, mode=2)
                feat = feat.detach()

            output = classifier(feat)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if args.local_rank == 0 and idx % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, idx, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()

        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        return top1.avg, top5.avg, losses.avg

    def validate(self, epoch, val_loader, model, classifier, criterion):
        time1 = time.time()
        args = self.args

        model.eval()
        classifier.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            end = time.time()
            for idx, (input, target) in enumerate(val_loader):
                input = input.float()
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                feat = model(x=input, mode=2)
                output = classifier(feat)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if args.local_rank == 0 and idx % args.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           idx, len(val_loader), batch_time=batch_time,
                           loss=losses, top1=top1, top5=top5))

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

        time2 = time.time()
        print('eval epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        return top1.avg, top5.avg, losses.avg
