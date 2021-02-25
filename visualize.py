import torchvision
from torch.utils.tensorboard import SummaryWriter


class visualizer():
    # to keep track of the steps for summary writer object
    def __init__(self, cfg, log_dir=None):
        super(visualizer, self).__init__()

        self.train_step = 0
        self.val_step = 0

        self.train_log_every = cfg['train_log_every']
        self.val_log_every = cfg['val_log_every']

        if log_dir != None:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = SummaryWriter()

    def log_train(self, args):
        self.train_step += 1
        step = self.train_step * self.train_log_every

        self.writer.add_scalar('train/loss',
                               args['running_loss'] / step,
                               step)

    def log_val(self, args):
        self.val_step += 1
        # step = self.val_step * self.val_log_every
        step = self.val_step

        self.writer.add_scalar('validation/loss',
                               args['val_loss'],
                               step)

        self.writer.add_scalar('validation/dice',
                               args['val_dice'],
                               step)

        grid_binary_pred = torchvision.utils.make_grid(args['binary_pred'][:, None, :, :])
        self.writer.add_image('validation/binary_precition',
                              grid_binary_pred,
                              step)

        grid_label = torchvision.utils.make_grid(args['label'][:, None, :, :])
        self.writer.add_image('validation/label',
                              grid_label,
                              step)

        if 'pred' in args.keys():
            grid_pred = torchvision.utils.make_grid(args['pred'][:, None, :, :])
            self.writer.add_image('validation/prediction',
                                  grid_pred,
                                  step)
