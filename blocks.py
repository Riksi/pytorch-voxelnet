import torch
import torch.nn as nn
from typing import List


class VFE_Layer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        A VFE layer class
        :param c_in: int, channel dimension of input
        :param c_out: int, the dimension of output after VFE, must be even
        """
        super(VFE_Layer, self).__init__()
        if out_channels % 2 != 0:
            raise ValueError(f'c_out must be even, got {out_channels}')
        self.units = out_channels // 2
        self.fcn = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=out_channels),
            nn.ReLU()
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    # TODO: how do switch training on/off
    def __call__(self, inputs, mask):
        """
        Call method of class
        :param inputs: Tensor [batch_size=B, out_dim=F, max_num_voxels=V, max_num_points=P]
        :param mask: TODO: what is this?
        :return: Tensor [B, c_out=C, V, P]
        """
        # [B, C//2, V, P]
        fcn_out = self.bn(self.fcn(inputs))
        # [B, C//2, V, 1]
        max_pool = torch.max(fcn_out, dim=-1, keepdim=True)
        tiled_max_pool = max_pool.repeat([1, 1, 1, fcn_out.size(-1)])
        # [B, C, V, P]
        output = torch.cat([fcn_out, tiled_max_pool], dim=1)
        # TODO: I think it broadcasts
        #mask = mask.repeat([1, 2*self.units, 1, 1,])
        return output * mask.float()


class VFE_Block(nn.Module):
    def __init__(self, in_channels:int,
                 vfe_out_dims:List[int],
                 final_dim:int,
                 sparse_shape):
        """

        :param in_channels: int, channel dimension of input
        :param vfe_out_dims: n-integer list of output dims of VFEs, where each must be even
        :param final_dim: int, dimension of layer dense layer after VFEs
        TODO: is it depth, height, width?
        :param sparse_shape: [depth, height, width] of voxel space
        """
        super(VFE_Block, self).__init__()

        self.vfe_out_dims = vfe_out_dims
        self.final_dim = final_dim
        self.sparse_shape = sparse_shape

        self.VFEs = nn.ModuleList(
            [VFE_Layer(dim_in, dim_out)
            for dim_in, dim_out
            in zip([in_channels] + vfe_out_dims[:-1], vfe_out_dims)]
        )

        self.final_fcn = nn.Sequential(
            nn.Linear(in_features=vfe_out_dims[-1], out_features=final_dim),
            nn.ReLU()
        )

    def __call_(self, inputs, voxel_coor_buffer, shape):
        """
        TODO: K?, T?
        :param inputs: Tensor [batch_size=B, out_dim=F, max_num_voxels=V, max_num_points=P]
        :param voxel_coor_buffer: int32 Tensor [batch_size, 4]
        :param shape: TODO [depth=D, height=H, width=W] ?
        :return: Tensor [B, C, depth, height, width]
        """
        vfe_out = inputs
        # [B, 1, V, P]
        mask = torch.ne(torch.max(inputs, dim=1, keepdim=True), 0)

        for i, vfe in enumerate(self.VFEs):
            vfe_out = vfe(vfe_out, mask)

        # [B, C, V, P]
        final = self.final_fcn(vfe_out)
        # [B, C, V]
        final = torch.max(final, dim=-1)

        # [B, C] -> [B, 1, 1, 1, C] -> [B, D, H, W, C]
        voxel_sparse = torch.zeros_like(final[..., 0])[:, None, None, None].repeat(
            [1, *shape, 1]
        )
        idx1, idx2, idx3, idx4 = voxel_coor_buffer.permute((1, 0))
        # [B, C, V] -> [B, V, C] scattered to [B, D, H, W, C]
        voxel_sparse[idx1, idx2, idx3, idx4] = final.permute((0, 2, 1))

        # [B, D, H, W, C] -> [B, C, D, H, W]
        return voxel_sparse.permute((0, 4, 1, 2, 3))


class ConvMiddleLayer(nn.Module):
    def __init__(self, in_channels, out_shape):
        """
        Convolutional Middle layer class
        :param in_channels: number of channels of input
        :param out_shape: dimensions of output [batch_size=B, new_channels=C, height=H, width=W]
        """

        self.out_shape = out_shape
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=64,
                               kernel_size=3, stride=(2, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=64,
                               kernel_size=3, stride=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=in_channels, out_channels=64,
                               kernel_size=3, stride=(2, 1, 1))

        self.bn1, self.bn2, self.bn3 = [nn.BatchNorm3d(num_features=64) for _ in range(3)]

        self.pad1 = torch.nn.ConstantPad3d(padding=(1, 1) * 3, value=0)
        self.pad2 = torch.nn.ConstantPad3d(padding=(0, 0) + (1, 1) * 2, value=0)
        self.pad3 = torch.nn.ConstantPad3d(padding=(1, 1) * 3, value=0)

        self.relu1, self.relu2, self.relu3 = [torch.nn.ReLU() for _ in range(3)]

    def __call__(self, inputs):
        """
        :param inputs: Tensor [batch_size=B, channels=C, depth=D, height=H, width=W]
        :return:
        """

        out = inputs

        for i in range(3):
            out = getattr(self, f'conv{i}')(out)
            out = getattr(self, f'bn{i}')(out)
            out = getattr(self, f'relu{i}')(out)
            out = getattr(self, f'pad{i}')(out)

        out = torch.reshape(out, self.out_shape)

        return out


class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors_per_cell):
        super(RPN, self).__init__()
        self.num_anchors_per_cell = num_anchors_per_cell
        BN = nn.BatchNorm2d

        self.conv1_block1, self.bn1_block1 = self._conv_layer(128, 3, 2, 1, c_in=in_channels), BN(128)
        for i in range(2, 5):
            setattr(self, f'conv{i}_block1', self._conv_layer(128, 3, 1, 1))
            setattr(self, f'bn{i}_block1', BN(128))

        self.conv1_block2, self.bn1_block2 = self._conv_layer(128, 3, 2, 1), BN(128)
        for i in range(2, 7):
            setattr(self, f'conv{i}_block2', self._conv_layer(128, 3, 1, 1))
            setattr(self, f'bn{i}_block2', BN(128))

        self.conv1_block3, self.bn1_block3 = self._conv_layer(256, 3, 2, 1, 1, c_in=128), BN(128)
        for i in range(2, 7):
            setattr(self, f'conv{i}_block3', self._conv_layer(256, 3, 1, 1))
            setattr(self, f'bn{i}_block3', BN(256))

        self.deconv_1, self.deconv_bn1 = self._deconv_layer(256, 3, 1, 1), BN(256)
        self.deconv_2, self.deconv_bn2 = self._deconv_layer(256, 2, 2, 0), BN(256)
        self.deconv_2, self.deconv_bn2 = self._deconv_layer(256, 4, 4, 0), BN(256)

        self.prob_map_conv = self._conv_layer(self.num_anchors_per_cell, 1, 1, 0, c_in=256 * 3)
        self.reg_map_conv = self._conv_layer(7 * self.num_anchors_per_cell, 1, 1, 0, c_in=256 * 3)

    def _conv_layer(self, c_out, kernel_size, stride, padding, c_in=None):
        if c_in is None:
            c_in = c_out
        return nn.Conv2d(out_channels=c_out,
                         in_channels=c_in,
                         padding=1,
                         kernel_size=3,
                         stride=stride)

    def _deconv_layer(self, c_out, kernel_size, stride, padding):
        return nn.ConvTranspose2d(in_channels=c_out,
                                  out_channels=c_out,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding)

    def block_conv(self, block_idx, x):
        i = 1
        while True:
            try:
                x = getattr(self, f'conv{i}_block{block_idx}')(x)
                x = getattr(self, f'bn{i}_block{block_idx}')(x)
                x = nn.functional.relu(x)
                i += 1
            except:
                break
        return x

    def deconv(self, idx, x):
        x = getattr(x, f'deconv_{idx}')(x)
        x = getattr(x, f'deconv_bn{idx}')(x)
        return nn.functional.relu(x)

    def __call__(self, x):
        # TODO: add assertion about shape

        deconvs = []

        for i in range(1, 4):
            x = self.block_conv(i, x)
            deconvs.append(self.deconv(i, x))

        x = torch.cat(deconvs, dim=1)

        prob_map = self.prob_map_conv(x)
        reg_map = self.reg_map_conv(x)

        # TODO: should these be transposed as in the code?
        #  should sigmoid be applied to reg_map

        return prob_map, reg_map











