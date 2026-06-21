import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_total, T_warmup, lr_min=0, last_epoch=-1):
        """
        Warmup + Cosine Annealing 学习率调度器
        Args:
            optimizer: 优化器
            T_total: 总的迭代次数
            T_warmup: 预热的迭代次数
            lr_min: 最小学习率
            last_epoch: 上一个 epoch 的索引
        """
        self.T_total = T_total
        self.T_warmup = T_warmup
        self.lr_min = lr_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 当前的迭代次数
        current_step = self.last_epoch + 1

        if current_step <= self.T_warmup:
            # Warmup 阶段：线性增加学习率
            return [base_lr * current_step / self.T_warmup for base_lr in self.base_lrs]
        else:
            # Cosine Annealing 阶段
            return [
                self.lr_min + (base_lr - self.lr_min) * 0.5 *
                (1 + math.cos(math.pi * (current_step - self.T_warmup) / (self.T_total - self.T_warmup)))
                for base_lr in self.base_lrs
            ]
