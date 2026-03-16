class Logger:
    def __init__(self, lr_scheduler,
                 summary_writer,
                 summary_freq=100,
                 start_step=0,
                 supervised=True
                 ):
        self.lr_scheduler = lr_scheduler
        self.total_steps = start_step
        self.running_loss = {}
        self.summary_writer = summary_writer
        self.summary_freq = summary_freq
        self.supervised = supervised

    def print_training_status(self, mode='train'):

        if self.supervised:
            print('step: %06d \t AEE: %.3f' % (self.total_steps, self.running_loss['AEE'] / self.summary_freq))
        else:
            print('step: %06d \t Gray: %.5f \t MrDGC: %.3f \t Total: %.5f' % 
                (self.total_steps, self.running_loss['Gray'] / self.summary_freq, self.running_loss['MrDGC'] /
                  self.summary_freq, self.running_loss['Total'] / self.summary_freq))

        for k in self.running_loss:
            self.summary_writer.add_scalar(mode + '/' + k,
                                           self.running_loss[k] / self.summary_freq, self.total_steps)
            self.running_loss[k] = 0.0

    def lr_summary(self):
        lr = self.lr_scheduler.get_last_lr()[0]
        self.summary_writer.add_scalar('lr', lr, self.total_steps)

    def push(self, metrics, mode='train'):
        self.total_steps += 1

        self.lr_summary()

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.summary_freq == 0:
            self.print_training_status(mode)
            self.running_loss = {}

    def write_dict(self, results):
        for key in results:
            tag = key.split('_')[0]
            tag = tag + '/' + key
            self.summary_writer.add_scalar(tag, results[key], self.total_steps)

    def close(self):
        self.summary_writer.close()
