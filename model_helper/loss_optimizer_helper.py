import torch
from collections import namedtuple

LossTuple = namedtuple('LossTuple',
                       ['total', 'reg', 'cls', 'cls_pos', 'cls_neg'])


class Loss:
    def __init__(self, params):
        self.global_batch_size = params["batch_size"]
        self.small_addon_for_BCE = params["small_addon_for_BCE"]
        self.alpha_bce = params["alpha_bce"]
        self.beta_bce = params["beta_bce"]

        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.smooth_l1 = torch.nn.L1Loss(reduction='none')

    def reg_loss_fn(self,
                    reg_target,
                    reg_pred,
                    pos_equal_one_reg,
                    pos_equal_one_sum):
        loss = self.smooth_l1(reg_target * pos_equal_one_reg,
                              reg_pred * pos_equal_one_reg) / pos_equal_one_sum
        return torch.mean(loss) * (1. / self.global_batch_size)

    def prob_loss_fn(self,
                     prob_pred,
                     pos_equal_one,
                     pos_equal_one_sum,
                     neg_equal_one,
                     neg_equal_one_sum):
        mask = torch.logical_or(pos_equal_one, neg_equal_one)

        prob_pred = prob_pred[mask]
        labels = pos_equal_one[mask]

        cls_losses = self.ce_loss(prob_pred, labels)
        cls_pos_losses = cls_losses[labels.bool()] / pos_equal_one_sum
        cls_neg_losses = cls_losses[(1 - labels).bool()] / neg_equal_one_sum

        cls_pos_loss = torch.sum(cls_pos_losses) * (1. / self.global_batch_size)
        cls_neg_loss = torch.sum(cls_neg_losses) * (1. / self.global_batch_size)

        cls_loss = self.alpha_bce * cls_pos_loss + self.beta_bce * cls_neg_loss

        return cls_loss, cls_pos_loss, cls_neg_loss

    def __call__(self, reg_pred, prob_pred,
                 targets,
                 pos_equal_one,
                 pos_equal_one_reg,
                 pos_equal_one_sum,
                 neg_equal_one,
                 neg_equal_one_sum):
        reg_loss = self.reg_loss_fn(targets, reg_pred, pos_equal_one_reg, pos_equal_one_sum)
        cls_loss, cls_pos_loss, cls_neg_loss = self.prob_loss_fn(prob_pred,
                                                                 pos_equal_one,
                                                                 pos_equal_one_sum,
                                                                 neg_equal_one,
                                                                 neg_equal_one_sum)
        loss = cls_loss + reg_loss

        return LossTuple(total=loss, cls=cls_loss, reg=reg_loss,
                         cls_pos=cls_pos_loss, cls_neg=cls_neg_loss)


class Optimizer:
    def __init__(self, model, params):
        boundaries = [80, 120]
        self.lr_cst = params["learning_rate"]
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr_cst)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=boundaries
        )

