import torch
import torch.nn as nn
from ...utils import box_coder_utils, box_utils, common_utils
from .point_head_template import PointHeadTemplate


class SECONDPointHead(PointHeadTemplate):
    """A simple point head for learning auxiliary tasks"""
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        if model_cfg.ENABLE:
            assert len(model_cfg.POINT_TASKS) > 0
            assert len(model_cfg.POINT_SRC) == len(model_cfg.POINT_SRC_DIM)
        self.point_cloud_range = model_cfg.POINT_CLOUD_RANGE
        self.voxel_size = model_cfg.VOXEL_SIZE

        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=model_cfg.POINT_SRC_UNIFIED_DIM,
            output_channels=num_class
        ) if 'cls' in model_cfg.POINT_TASKS else None

        self.part_reg_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.PART_FC,
            input_channels=model_cfg.POINT_SRC_UNIFIED_DIM,
            output_channels=3
        ) if 'part' in model_cfg.POINT_TASKS else None

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        ) if 'reg' in model_cfg.POINT_TASKS else None
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=model_cfg.POINT_SRC_UNIFIED_DIM,
            output_channels=3 if model_cfg.POINT_REGRESS_TO == 'center' else self.box_coder.code_size
        ) if 'reg' in model_cfg.POINT_TASKS else None

        self.src_dim_matcher = nn.ModuleList([
            nn.Linear(model_cfg.POINT_SRC_DIM[i], model_cfg.POINT_SRC_UNIFIED_DIM, bias=False)
            for i in range(len(model_cfg.POINT_SRC))
        ])

    @torch.no_grad()
    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=True, ret_box_labels=(self.box_layers is not None)
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss = 0
        if self.cls_layers is not None:
            point_loss_cls, tb_dict = self.get_cls_layer_loss(tb_dict)
            point_loss += point_loss_cls

        if self.part_reg_layers is not None:
            point_loss_part, tb_dict = self.get_part_layer_loss(tb_dict)
            point_loss += point_loss_part

        if self.box_layers is not None:
            point_loss_box, tb_dict = self.get_box_layer_loss(tb_dict)
            point_loss += point_loss_box
        return point_loss, tb_dict

    def voxels2pts(self, sparse_tensor, stride, batch_size):
        """
        Get voxels' center coord
        Args:
            sparse_tensor (SparseConvTensor):
            stride (int):
            batch_size (int):

        Returns:
            vox_xyz: (N, 3)
            vox_batch_cnt: (B) - [N_0, N_1, ...] | sum(N_i) = N
            vox_feat: (N, C)
        """
        cur_coords = sparse_tensor.indices
        vox_xyz = common_utils.get_voxel_centers(
            cur_coords[:, 1:4],
            downsample_times=stride,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )  # (n_vox, 3)

        vox_batch_cnt = vox_xyz.new_zeros(batch_size).int()  # (bs) - [n_vox_0, n_vox_1, ...]; sum_i (n_vox_i) = n_vox
        for bs_idx in range(batch_size):
            vox_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

        return vox_xyz, vox_batch_cnt, sparse_tensor.features

    def forward(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if 'unet_out' not in self.model_cfg.POINT_SRC:
            list_feat, list_batch_cnt = [], []
            for i, src in enumerate(self.model_cfg.POINT_SRC):
                src_tensor = batch_dict['multi_scale_3d_features'][src]
                src_stride = batch_dict['multi_scale_3d_strides'][src]
                vox_xyz, vox_batch_cnt, vox_feat = self.voxels2pts(src_tensor, src_stride, batch_size)
                # match vox feat across different src
                vox_feat = self.src_dim_matcher[i](vox_feat)
                # store
                feat = torch.cat([vox_xyz, vox_feat], dim=1)  # (N, 3 + C)
                list_feat.append(feat)
                list_batch_cnt.append(vox_batch_cnt)

            # merge points from different sources
            merge_feat, merge_batch_idx, merge_batch_cnt = merge_list_feat(list_feat, list_batch_cnt)
            point_coords = merge_feat[:, :3]
            point_features = merge_feat[:, 3:].contiguous()

            n_points_tot = merge_batch_cnt.sum()
            assert point_coords.shape == (n_points_tot, 3)
            assert point_features.shape == (n_points_tot, self.model_cfg.POINT_SRC_UNIFIED_DIM)

            point_coords = torch.cat([merge_batch_idx.reshape(-1, 1).float(), point_coords], dim=1).contiguous()
            batch_dict['point_coords'] = point_coords  # (N_tot, 4) - [batch_idx, x, y, z]
            batch_dict['point_features'] = point_features  # (N_tot, C)
        else:
            assert self.model_cfg.POINT_SRC == ['unet_out']
            point_features = batch_dict['point_features']  # x_up1, stride=1, 16 channels
            point_features = self.src_dim_matcher[0](point_features)

        point_cls_preds = self.cls_layers(point_features) if self.cls_layers is not None else None  # (N_tot, num_class)
        point_part_preds = self.part_reg_layers(point_features) if self.part_reg_layers is not None else None  # (N_tot, 3)
        point_box_preds = self.box_layers(point_features) if self.box_layers is not None else None  # (N_tot, 3 or box_code_size)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
            'point_part_preds': point_part_preds,
            'point_box_preds': point_box_preds
        }

        if point_cls_preds is not None:
            point_cls_scores = torch.sigmoid(point_cls_preds)
            batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)
        if point_part_preds is not None:
            point_part_offset = torch.sigmoid(point_part_preds)
            batch_dict['point_part_offset'] = point_part_offset

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_part_labels'] = targets_dict.get('point_part_labels')
            ret_dict['point_box_labels'] = targets_dict.get('point_box_labels')
            if ret_dict['point_box_labels'] is not None and self.model_cfg.POINT_REGRESS_TO == 'center':
                ret_dict['point_box_labels'] = ret_dict['point_box_labels'][:, :3]

        self.forward_ret_dict = ret_dict
        return batch_dict


def merge_list_feat(list_feat, list_batch_cnt):
    """
    Merge a list of feat into a single feat

    Args:
        list_feat (list): each element is (N, C)
        list_batch_cnt (list): each element is (B) - [N0, N1, ...]; sum(Ni) == N

    Returns:
        merge_feat (torch.Tensor): (M, C)
        batch_idx (torch.Tensor): (M)
        merge_batch_cnt (torch.Tensor): (B) - [M0, M1, ...]
    """
    assert isinstance(list_feat, list) and isinstance(list_batch_cnt, list)
    assert len(list_feat) == len(list_batch_cnt)
    batch_size = list_batch_cnt[0].shape[0]
    for bi in range(len(list_batch_cnt)):
        assert list_batch_cnt[bi].shape[0] == batch_size

    merge_feat = []
    merge_batch_idx = []
    for bi in range(batch_size):
        for feat, batch_cnt in zip(list_feat, list_batch_cnt):
            start_idx = torch.sum(batch_cnt[:bi]) if bi > 0 else 0
            end_idx = start_idx + batch_cnt[bi]
            # extract feat in this data sample
            merge_feat.append(feat[start_idx: end_idx])  # (Ni, C)
            merge_batch_idx.append(feat.new_zeros(batch_cnt[bi]).fill_(bi))  # (Ni)

    merge_feat = torch.cat(merge_feat, dim=0)
    merge_batch_idx = torch.cat(merge_batch_idx, dim=0)

    merge_batch_cnt = torch.cat([batch_cnt.unsqueeze(1) for batch_cnt in list_batch_cnt], dim=1).sum(dim=1)
    return merge_feat, merge_batch_idx, merge_batch_cnt
