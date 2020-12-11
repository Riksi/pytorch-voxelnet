import torch
import torch.nn as nn
from blocks import RPN, ConvMiddleLayer, VFE_Block
from model_helper.loss_optimizer_helper import Loss, Optimizer


def validate_step(model,
                  feature_buffer,
                  coordinate_buffer,
                  targets,
                  pos_equal_one,
                  pos_equal_one_reg,
                  pos_equal_one_sum,
                  neg_equal_one,
                  neg_equal_one_sum):

    p_map, r_map = model(feature_buffer=feature_buffer,
                         coordinate_buffer=coordinate_buffer)
    loss = model.loss(r_map, p_map,
                      targets,
                      pos_equal_one,
                      pos_equal_one_reg,
                      pos_equal_one_sum,
                      neg_equal_one,
                      neg_equal_one_sum)

    return loss


def train_step(model,
               optimizer,
               feature_buffer,
               coordinate_buffer,
               targets,
               pos_equal_one,
               pos_equal_one_reg,
               pos_equal_one_sum,
               neg_equal_one,
               neg_equal_one_sum):

    optimizer.optimizer.zero_grad()
    p_map, r_map = model(feature_buffer=feature_buffer,
                         coordinate_buffer=coordinate_buffer)
    loss = model.loss(r_map, p_map,
                      targets,
                      pos_equal_one,
                      pos_equal_one_reg,
                      pos_equal_one_sum,
                      neg_equal_one,
                      neg_equal_one_sum)

    loss.total.backward()
    nn.utils.clip_grad_norm_(model.parameters(),
                             model.params["max_gradient_norm"])

    optimizer.optimizer.step()
    optimizer.scheduler.step()
    return loss


class Model(nn.Module):
    def __init__(self, cfg, params, strategy, *args, **kwargs):
        super(Model, self).__init__()
        self.strategy = strategy
        n_replicas = self.strategy.num_replicas_in_sync
        self.params = params
        self.cfg = cfg
        self.vfe_block = VFE_Block(in_channels=cfg.VFE_IN_DIMS,
                                   vfe_out_dims=cfg.VFE_OUT_DIMS,
                                   final_dim=cfg.VFE_FINAL_DIM,
                                   sparse_shape=cfg.GRID_SIZE)
        self.conv_middle = ConvMiddleLayer(
            in_channels=cfg.VFE_FINAL_DIM,
            out_shape=(params['batch_size'] // n_replicas, -1, *cfg.GRID_SIZE[1:])
        )

        self.rpn = RPN(
            in_channels=64,
            num_anchors_per_cell=cfg.NUM_ANCHORS_PER_CELL
        )

        self.loss = Loss(self.parameters())

    def forward(self, batch, *args, **kwargs):
        if batch is None:
            if not ("coordinate_buffer" in kwargs
                 and "feature_buffer" in kwargs):
                raise ValueError("You must provide a `batch` object or a dict with keys"
                                 "`coordinate_buffer` and `feature_buffer` corresponding"
                                 "to tensors")
            batch = kwargs

        n_replicas = self.strategy.num_replicas_in_sync
        shape = [self.params["batch_size"] // n_replicas,
                 *self.vfe_block.sparse_shape,
                 self.vfe_block.final_dim]
        output = self.vfe_block(batch["feature_buffer"],
                                batch["coordinate_buffer"],
                                shape)
        output = self.conv_middle(output)
        prob_map, reg_map = self.rpn(output)
        return prob_map, reg_map











