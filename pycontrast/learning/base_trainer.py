from __future__ import print_function

import os
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger


class BaseTrainer(object):
    """class for BaseTrainer"""
    def __init__(self, args):
        self.args = args
        self.local_group = None
        self.logger = None

    def init_ddp_environment(self, gpu, ngpus_per_node):
        """
        Args:
          gpu: current gpu id
          ngpus_per_node: num of process/gpus per node
        """
        self.args.gpu = gpu
        self.args.ngpus_per_node = ngpus_per_node
        self.args.node_rank = self.args.rank
        self.args.local_rank = gpu
        self.args.local_center = self.args.rank * ngpus_per_node

        torch.cuda.set_device(gpu)
        cudnn.benchmark = True

        if self.args.gpu is not None:
            print("Use GPU: {} for training".format(self.args.gpu))

        if self.args.distributed:
            if self.args.multiprocessing_distributed:
                self.args.rank = self.args.rank * ngpus_per_node + gpu
            os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
            dist.init_process_group(
                backend=self.args.dist_backend, init_method=self.args.dist_url,
                world_size=self.args.world_size, rank=self.args.rank)

        # setup local group on each node, for ShuffleBN
        local_groups = []
        for i in range(0, self.args.world_size // ngpus_per_node):
            gp = torch.distributed.new_group(
                ranks=list(range(i * ngpus_per_node, (i + 1) * ngpus_per_node)),
                backend=self.args.dist_backend)
            local_groups.append(gp)

        local_group = local_groups[self.args.rank // ngpus_per_node]
        if self.args.local_rank == 0:
            print("node_rank:", self.args.node_rank)
            print("local_center:", self.args.local_center)
            print("local group size:", dist.get_world_size(local_group))

        self.local_group = local_group

    def init_tensorboard_logger(self):
        args = self.args
        if args.rank == 0:
            self.logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    def adjust_learning_rate(self, optimizer, epoch):
        args = self.args
        lr = args.learning_rate
        if args.cosine:
            eta_min = lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (lr - eta_min) * (
                    1 + math.cos(math.pi * epoch / args.epochs)) / 2
        else:
            steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
            if steps > 0:
                lr = lr * (args.lr_decay_rate ** steps)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def warmup_learning_rate(self, epoch, batch_id, total_batches, optimizer):
        args = self.args
        if args.warm and epoch <= args.warm_epochs:
            p = (batch_id + (epoch - 1) * total_batches) / \
                (args.warm_epochs * total_batches)
            lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
