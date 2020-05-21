import os
import math
from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--aug', default='A', type=str,
                            help='data augmentation for training')
        parser.add_argument('--beta', type=float, default=0.5,
                            help='balance between Jigsaw and InsDis')
        parser.add_argument('--warm', action='store_true',
                            help='add warm-up setting')
        parser.add_argument('--amp', action='store_true',
                            help='using mixed precision')
        parser.add_argument('--opt_level', type=str, default='O2',
                            choices=['O1', 'O2'])

        return parser

    def modify_options(self, opt):
        opt = self.override_options(opt)

        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

        # set up saving name
        opt.model_name = '{}_{}_{}_Jig_{}_{}_aug_{}_{}_{}'.format(
            opt.method, opt.arch, opt.modal, opt.jigsaw, opt.mem,
            opt.aug, opt.head, opt.nce_t
        )
        if opt.amp:
            opt.model_name = '{}_amp_{}'.format(opt.model_name, opt.opt_level)
        if opt.cosine:
            opt.model_name = '{}_cosine'.format(opt.model_name)

        # warm-up for large-batch training, e.g. 1024 with multiple nodes
        if opt.batch_size > 256:
            opt.warm = True
        if opt.warm:
            opt.model_name = '{}_warm'.format(opt.model_name)
            opt.warmup_from = 0.01
            if opt.epochs > 500:
                opt.warm_epochs = 10
            else:
                opt.warm_epochs = 5
            if opt.cosine:
                eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
                opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                            1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
            else:
                opt.warmup_to = opt.learning_rate

        # create folders
        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)
        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        return opt
