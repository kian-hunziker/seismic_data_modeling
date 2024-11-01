import torch


class DoubleLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, num_warmup_steps, num_cooldown_steps, last_epoch=-1):
        self.warmup_steps = num_warmup_steps
        self.cooldown_steps = num_cooldown_steps

        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step):
        # First warmup phase
        if step < self.warmup_steps:
            return step / self.warmup_steps
        # First cooldown to zero
        elif step < self.warmup_steps + self.cooldown_steps:
            return max(0.0, (self.warmup_steps + self.cooldown_steps - step) / self.cooldown_steps)
        # First plateau (at zero)
        elif step < 2 * self.warmup_steps + self.cooldown_steps:
            return (step - self.warmup_steps - self.cooldown_steps) / self.warmup_steps

        else:
            return max(0.0, (2 * self.warmup_steps + 2 * self.cooldown_steps - step) / self.cooldown_steps)