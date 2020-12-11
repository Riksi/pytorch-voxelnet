import torch
import torch.nn as nn
from blocks import RPN, ConvMiddleLayer, VFE_Block
from model_helper.loss_optimizer_helper import Loss, Optimizer


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

    def call(self, batch, *args, **kwargs):
        if batch is None:
            if not ("" in kwargs
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






