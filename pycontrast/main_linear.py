"""
DDP training for Linear Probing
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from options.test_options import TestOptions
from learning.linear_trainer import LinearTrainer
from networks.build_backbone import build_model
from networks.build_linear import build_linear
from datasets.util import build_linear_loader


def main():
    args = TestOptions().parse()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        raise NotImplementedError('Currently only DDP training')


def main_worker(gpu, ngpus_per_node, args):

    # initialize trainer and ddp environment
    trainer = LinearTrainer(args)
    trainer.init_ddp_environment(gpu, ngpus_per_node)

    # build encoder and classifier
    model, _ = build_model(args)
    classifier = build_linear(args)

    # build dataset
    train_loader, val_loader, train_sampler = \
        build_linear_loader(args, ngpus_per_node)

    # build criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(classifier.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # load pre-trained ckpt for encoder
    model = trainer.load_encoder_weights(model)

    # wrap up models
    model, classifier = trainer.wrap_up(model, classifier)

    # check and resume a classifier
    start_epoch = trainer.resume_model(classifier, optimizer)

    # init tensorboard logger
    trainer.init_tensorboard_logger()

    # routine
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        trainer.adjust_learning_rate(optimizer, epoch)

        outs = trainer.train(epoch, train_loader, model, classifier,
                             criterion, optimizer)

        # log to tensorbard
        trainer.logging(epoch, outs, optimizer.param_groups[0]['lr'], train=True)

        # evaluation and logging
        if args.rank % ngpus_per_node == 0:
            outs = trainer.validate(epoch, val_loader, model,
                                    classifier, criterion)
            trainer.logging(epoch, outs, train=False)

        # saving model
        trainer.save(classifier, optimizer, epoch)


if __name__ == '__main__':
    main()
