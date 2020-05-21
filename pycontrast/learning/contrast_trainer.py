from __future__ import print_function

import os
import sys
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .util import AverageMeter, accuracy
from .base_trainer import BaseTrainer

try:
    from apex import amp, optimizers
except ImportError:
    pass


class ContrastTrainer(BaseTrainer):
    """trainer for contrastive pretraining"""
    def __init__(self, args):
        super(ContrastTrainer, self).__init__(args)

    def logging(self, epoch, logs, lr):
        """ logging to tensorboard

        Args:
          epoch: training epoch
          logs: loss and accuracy
          lr: learning rate
        """
        args = self.args
        if args.rank == 0:
            self.logger.log_value('loss', logs[0], epoch)
            self.logger.log_value('acc', logs[1], epoch)
            self.logger.log_value('jig_loss', logs[2], epoch)
            self.logger.log_value('jig_acc', logs[3], epoch)
            self.logger.log_value('learning_rate', lr, epoch)

    def wrap_up(self, model, model_ema, optimizer):
        """Wrap up models with apex and DDP

        Args:
          model: model
          model_ema: momentum encoder
          optimizer: optimizer
        """
        args = self.args

        model.cuda(args.gpu)
        if isinstance(model_ema, torch.nn.Module):
            model_ema.cuda(args.gpu)

        # to amp model if needed
        if args.amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.opt_level
            )
            if isinstance(model_ema, torch.nn.Module):
                model_ema = amp.initialize(
                    model_ema, opt_level=args.opt_level
                )
        # to distributed data parallel
        model = DDP(model, device_ids=[args.gpu])

        if isinstance(model_ema, torch.nn.Module):
            self.momentum_update(model.module, model_ema, 0)

        return model, model_ema, optimizer

    def broadcast_memory(self, contrast):
        """Synchronize memory buffers

        Args:
          contrast: memory.
        """
        if self.args.modal == 'RGB':
            dist.broadcast(contrast.memory, 0)
        else:
            dist.broadcast(contrast.memory_1, 0)
            dist.broadcast(contrast.memory_2, 0)

    def resume_model(self, model, model_ema, contrast, optimizer):
        """load checkpoint"""
        args = self.args
        start_epoch = 1
        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                start_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['model'])
                contrast.load_state_dict(checkpoint['contrast'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if isinstance(model_ema, torch.nn.Module):
                    model_ema.load_state_dict(checkpoint['model_ema'])
                if args.amp:
                    amp.load_state_dict(checkpoint['amp'])
                print("=> loaded successfully '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                del checkpoint
                torch.cuda.empty_cache()
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        return start_epoch

    def save(self, model, model_ema, contrast, optimizer, epoch):
        """save model to checkpoint"""
        args = self.args
        if args.local_rank == 0:
            # saving the model to each instance
            print('==> Saving...')
            state = {
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if isinstance(model_ema, torch.nn.Module):
                state['model_ema'] = model_ema.state_dict()
            if args.amp:
                state['amp'] = amp.state_dict()
            save_file = os.path.join(args.model_folder, 'current.pth')
            torch.save(state, save_file)
            if epoch % args.save_freq == 0:
                save_file = os.path.join(
                    args.model_folder, 'ckpt_epoch_{}.pth'.format(epoch))
                torch.save(state, save_file)
            # help release GPU memory
            del state

    def train(self, epoch, train_loader, model, model_ema, contrast,
              criterion, optimizer):
        """one epoch training"""
        args = self.args
        model.train()

        time1 = time.time()
        if args.mem == 'moco':
            outs = self._train_moco(epoch, train_loader, model, model_ema,
                                    contrast, criterion, optimizer)
        else:
            outs = self._train_mem(epoch, train_loader, model,
                                   contrast, criterion, optimizer)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        return outs

    @staticmethod
    def _global_gather(x):
        all_x = [torch.ones_like(x)
                 for _ in range(dist.get_world_size())]
        dist.all_gather(all_x, x, async_op=False)
        return torch.cat(all_x, dim=0)

    def _shuffle_bn(self, x, model_ema):
        """ Shuffle BN implementation

        Args:
          x: input image on each GPU/process
          model_ema: momentum encoder on each GPU/process
        """
        args = self.args
        local_gp = self.local_group
        bsz = x.size(0)

        # gather x locally for each node
        node_x = [torch.ones_like(x)
                  for _ in range(dist.get_world_size(local_gp))]
        dist.all_gather(node_x, x.contiguous(),
                        group=local_gp, async_op=False)
        node_x = torch.cat(node_x, dim=0)

        # shuffle bn
        shuffle_ids = torch.randperm(
            bsz * dist.get_world_size(local_gp)).cuda()
        reverse_ids = torch.argsort(shuffle_ids)
        dist.broadcast(shuffle_ids, 0)
        dist.broadcast(reverse_ids, 0)

        this_ids = shuffle_ids[args.local_rank*bsz:(args.local_rank+1)*bsz]
        with torch.no_grad():
            this_x = node_x[this_ids]
            if args.jigsaw:
                k = model_ema(this_x, x_jig=None, mode=1)
            else:
                k = model_ema(this_x, mode=1)

        # globally gather k
        all_k = self._global_gather(k)

        # unshuffle bn
        node_id = args.node_rank
        ngpus = args.ngpus_per_node
        node_k = all_k[node_id*ngpus*bsz:(node_id+1)*ngpus*bsz]
        this_ids = reverse_ids[args.local_rank*bsz:(args.local_rank+1)*bsz]
        k = node_k[this_ids]

        return k, all_k

    @staticmethod
    def _compute_loss_accuracy(logits, target, criterion):
        """
        Args:
          logits: a list of logits, each with a contrastive task
          target: contrastive learning target
          criterion: typically nn.CrossEntropyLoss
        """
        losses = [criterion(logit, target) for logit in logits]

        def acc(l, t):
            acc1 = accuracy(l, t)
            return acc1[0]

        accuracies = [acc(logit, target) for logit in logits]

        return losses, accuracies

    def _train_moco(self, epoch, train_loader, model, model_ema, contrast,
                    criterion, optimizer):
        """
        MoCo encoder style training. This needs two forward passes,
        one for normal encoder, and one for moco encoder
        """
        args = self.args
        model.train()
        model_ema.eval()

        def set_bn_train(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.train()
        model_ema.apply(set_bn_train)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        loss_jig_meter = AverageMeter()
        acc_jig_meter = AverageMeter()

        end = time.time()
        for idx, data in enumerate(train_loader):
            data_time.update(time.time() - end)

            inputs = data[0].float().cuda(args.gpu, non_blocking=True)
            bsz = inputs.size(0)

            # warm-up learning rate
            self.warmup_learning_rate(
                epoch, idx, len(train_loader), optimizer)

            # split into two crops
            x1, x2 = torch.split(inputs, [3, 3], dim=1)

            # shuffle BN for momentum encoder
            k, all_k = self._shuffle_bn(x2, model_ema)

            # loss and metrics
            if args.jigsaw:
                inputs_jig = data[2].float().cuda(args.gpu, non_blocking=True)
                bsz, m, c, h, w = inputs_jig.shape
                inputs_jig = inputs_jig.view(bsz * m, c, h, w)
                q, q_jig = model(x1, inputs_jig)
                if args.modal == 'CMC':
                    q1, q2 = torch.chunk(q, 2, dim=1)
                    q1_jig, q2_jig = torch.chunk(q_jig, 2, dim=1)
                    k1, k2 = torch.chunk(k, 2, dim=1)
                    all_k1, all_k2 = torch.chunk(all_k, 2, dim=1)
                    output = contrast(q1, k1, q2, k2, q2_jig, q1_jig,
                                      all_k1, all_k2)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = (1 - args.beta) * (losses[0] + losses[1]) + \
                        args.beta * (losses[2] + losses[3])
                    update_loss = 0.5 * (losses[0] + losses[1])
                    update_acc = 0.5 * (accuracies[0] + accuracies[1])
                    update_loss_jig = 0.5 * (losses[2] + losses[3])
                    update_acc_jig = 0.5 * (accuracies[2] + accuracies[3])
                else:
                    output = contrast(q, k, q_jig, all_k)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = (1 - args.beta) * losses[0] + \
                        args.beta * losses[1]
                    update_loss = losses[0]
                    update_acc = accuracies[0]
                    update_loss_jig = losses[1]
                    update_acc_jig = accuracies[1]
            else:
                q = model(x1)
                if args.modal == 'CMC':
                    q1, q2 = torch.chunk(q, 2, dim=1)
                    k1, k2 = torch.chunk(k, 2, dim=1)
                    all_k1, all_k2 = torch.chunk(all_k, 2, dim=1)
                    output = contrast(q1, k1, q2, k2,
                                      all_k1=all_k1,
                                      all_k2=all_k2)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = losses[0] + losses[1]
                    update_loss = 0.5 * (losses[0] + losses[1])
                    update_acc = 0.5 * (accuracies[0] + accuracies[1])
                    update_loss_jig = torch.tensor([0.0])
                    update_acc_jig = torch.tensor([0.0])
                else:
                    output = contrast(q, k, all_k=all_k)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = losses[0]
                    update_loss = losses[0]
                    update_acc = accuracies[0]
                    update_loss_jig = torch.tensor([0.0])
                    update_acc_jig = torch.tensor([0.0])

            # backward
            optimizer.zero_grad()
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # update meters
            loss_meter.update(update_loss.item(), bsz)
            loss_jig_meter.update(update_loss_jig.item(), bsz)
            acc_meter.update(update_acc[0], bsz)
            acc_jig_meter.update(update_acc_jig[0], bsz)

            # update momentum encoder
            self.momentum_update(model.module, model_ema, args.alpha)

            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if args.local_rank == 0:
                if (idx + 1) % args.print_freq == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'l_I {loss.val:.3f} ({loss.avg:.3f})\t'
                          'a_I {acc.val:.3f} ({acc.avg:.3f})\t'
                          'l_J {loss_jig.val:.3f} ({loss_jig.avg:.3f})\t'
                          'a_J {acc_jig.val:.3f} ({acc_jig.avg:.3f})'.format(
                           epoch, idx + 1, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=loss_meter, acc=acc_meter,
                           loss_jig=loss_jig_meter, acc_jig=acc_jig_meter))
                    sys.stdout.flush()

        return loss_meter.avg, acc_meter.avg, loss_jig_meter.avg, acc_jig_meter.avg

    def _train_mem(self, epoch, train_loader, model, contrast,
                   criterion, optimizer):
        """
        Training based on memory bank mechanism. Only one forward pass.
        """
        args = self.args
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        loss_jig_meter = AverageMeter()
        acc_jig_meter = AverageMeter()

        end = time.time()
        for idx, data in enumerate(train_loader):
            data_time.update(time.time() - end)

            inputs = data[0].float().cuda(args.gpu, non_blocking=True)
            index = data[1].cuda(args.gpu, non_blocking=True)
            bsz = inputs.size(0)

            # warm-up learning rate
            self.warmup_learning_rate(
                epoch, idx, len(train_loader), optimizer)

            # compute feature
            if args.jigsaw:
                inputs_jig = data[2].float().cuda(args.gpu, non_blocking=True)
                bsz, m, c, h, w = inputs_jig.shape
                inputs_jig = inputs_jig.view(bsz * m, c, h, w)
                f, f_jig = model(inputs, inputs_jig)
            else:
                f = model(inputs)

            # gather all feature and index
            all_f = self._global_gather(f)
            all_index = self._global_gather(index)

            # loss and metrics
            if args.jigsaw:
                if args.modal == 'CMC':
                    f1, f2 = torch.chunk(f, 2, dim=1)
                    f1_jig, f2_jig = torch.chunk(f_jig, 2, dim=1)
                    all_f1, all_f2 = torch.chunk(all_f, 2, dim=1)
                    output = contrast(f1, f2, index, f2_jig, f1_jig,
                                      all_f1, all_f2, all_index)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = (1 - args.beta) * (losses[0] + losses[1]) + \
                        args.beta * (losses[2] + losses[3])
                    update_loss = 0.5 * (losses[0] + losses[1])
                    update_acc = 0.5 * (accuracies[0] + accuracies[1])
                    update_loss_jig = 0.5 * (losses[2] + losses[3])
                    update_acc_jig = 0.5 * (accuracies[2] + accuracies[3])
                else:
                    output = contrast(f, index, f_jig, all_f, all_index)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = (1 - args.beta) * losses[0] + \
                        args.beta * losses[1]
                    update_loss = losses[0]
                    update_acc = accuracies[0]
                    update_loss_jig = losses[1]
                    update_acc_jig = accuracies[1]
            else:
                if args.modal == 'CMC':
                    f1, f2 = torch.chunk(f, 2, dim=1)
                    all_f1, all_f2 = torch.chunk(all_f, 2, dim=1)
                    output = contrast(f1, f2, index, None, None,
                                      all_f1, all_f2, all_index)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = losses[0] + losses[1]
                    update_loss = 0.5 * (losses[0] + losses[1])
                    update_acc = 0.5 * (accuracies[0] + accuracies[1])
                    update_loss_jig = torch.tensor([0.0])
                    update_acc_jig = torch.tensor([0.0])
                else:
                    output = contrast(f, index, None, all_f, all_index)
                    losses, accuracies = self._compute_loss_accuracy(
                        logits=output[:-1], target=output[-1],
                        criterion=criterion)
                    loss = losses[0]
                    update_loss = losses[0]
                    update_acc = accuracies[0]
                    update_loss_jig = torch.tensor([0.0])
                    update_acc_jig = torch.tensor([0.0])

            # backward
            optimizer.zero_grad()
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # update meters
            loss_meter.update(update_loss.item(), bsz)
            loss_jig_meter.update(update_loss_jig.item(), bsz)
            acc_meter.update(update_acc[0], bsz)
            acc_jig_meter.update(update_acc_jig[0], bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if args.local_rank == 0:
                if (idx + 1) % args.print_freq == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'l_I {loss.val:.3f} ({loss.avg:.3f})\t'
                          'a_I {acc.val:.3f} ({acc.avg:.3f})\t'
                          'l_J {loss_jig.val:.3f} ({loss_jig.avg:.3f})\t'
                          'a_J {acc_jig.val:.3f} ({acc_jig.avg:.3f})'.format(
                           epoch, idx + 1, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=loss_meter, acc=acc_meter,
                           loss_jig=loss_jig_meter, acc_jig=acc_jig_meter))
                    sys.stdout.flush()

        return loss_meter.avg, acc_meter.avg, loss_jig_meter.avg, acc_jig_meter.avg

    @staticmethod
    def momentum_update(model, model_ema, m):
        """ model_ema = m * model_ema + (1 - m) model """
        for p1, p2 in zip(model.parameters(), model_ema.parameters()):
            p2.data.mul_(m).add_(1 - m, p1.detach().data)
