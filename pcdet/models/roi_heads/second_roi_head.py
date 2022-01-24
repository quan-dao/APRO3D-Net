import torch
import torch.nn as nn
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from .attention_v2 import StackMetaFormers


class SECONDAttnHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.point_cloud_range = model_cfg.POINT_CLOUD_RANGE
        self.voxel_size = model_cfg.VOXEL_SIZE
        self.n_key_feat = 1  # ROI center

        # ===============
        # RCNN Head
        pre_channel = model_cfg.ATTENTION.DIM * self.n_key_feat
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class,
            fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        # =====
        # Pooling
        self.roipoint_pool3d_layers = nn.ModuleList([
            roipoint_pool3d_utils.RoIPointPool3d(
                num_sampled_points=n_pts,
                pool_extra_width=self.model_cfg.ROI_POINT_POOL.POOL_EXTRA_WIDTH
            ) for n_pts in model_cfg.ROI_POINT_POOL.N_POINTS_PER_SRC
        ])

        # ===============
        # ATTENTION
        self.src_dim_matcher = nn.ModuleList([
            nn.Linear(model_cfg.ROI_POINT_POOL.POOL_SRC_DIM[i], model_cfg.ATTENTION.DIM, bias=False)
            for i in range(len(model_cfg.ROI_POINT_POOL.POOL_SRC))
        ])

        self.roi_feat_initializer = nn.ModuleList([
            nn.Embedding(model_cfg.N_CLASS, model_cfg.ATTENTION.DIM) for i in range(self.n_key_feat)
        ])

        if model_cfg.ATTENTION.POSITION_ENCODING_MODE == 'center_diff':
            self.d_init_pos_enc = 3
        elif model_cfg.ATTENTION.POSITION_ENCODING_MODE == 'center_and_corners_diff':
            self.d_init_pos_enc = 27  # 1 center & 8 corners
        else:
            raise ValueError("Invalid option, choose from ('center_diff', 'center_and_corners_diff')")

        attn_cfg = model_cfg.ATTENTION
        if not attn_cfg.SEQUENTIAL:
            self.attn_stack = StackMetaFormers(
                n_transformers=attn_cfg.NUM_TRANSFORMERS,
                d=attn_cfg.DIM,
                d_pos_emb=self.d_init_pos_enc,
                hidden_scale=attn_cfg.HIDDEN_SCALE
            )
        else:
            # Pool sequentially from lowest resolution feat map to highest
            n_tf_per_stack = len(model_cfg.ROI_POINT_POOL.POOL_SRC)
            all_drop_path = torch.linspace(0, 0.3, steps=n_tf_per_stack * attn_cfg.N_REPETITION).numpy().tolist()
            self.attn_stack = nn.ModuleList()
            for i in range(attn_cfg.N_REPETITION):
                stack = StackMetaFormers(
                    n_transformers=n_tf_per_stack,
                    d=attn_cfg.DIM,
                    d_pos_emb=self.d_init_pos_enc,
                    drop_path_probs=all_drop_path[i * n_tf_per_stack: (i + 1) * n_tf_per_stack]
                )
                self.attn_stack.append(stack)

        # ===============
        self.init_weights('xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def voxels2pts(self, sparse_tensor, stride, batch_size, output_format='batch'):
        """
        Get voxels' center coord
        Args:
            sparse_tensor (SparseConvTensor):
            stride (int):
            batch_size (int):
            output_format (str): if 'batch' -> (B, N, 3); else xyz & batch_cnt

        Returns:
            vox_xyz: (B, N, 3) or (N, 3)
            vox_feat: (B, N, C) or (N, C)
            vox_batch_cnt: (B) - [N_0, N_1, ...] | sum(N_i) = N
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

        if output_format != 'batch':
            return vox_xyz, vox_batch_cnt, sparse_tensor.features

        max_batch_cnt = torch.max(vox_batch_cnt)
        # pad sample with less voxels than max_batch_cnt with coord outside of point_cloud_range so that
        # these dummy coord don't get pooled by any ROIs
        out_xyz = torch.tensor([80, 50, 2],
                               device=vox_xyz.device).float().reshape(1, 1, 3).repeat(batch_size, max_batch_cnt, 1)
        out_feat = out_xyz.new_zeros((batch_size, max_batch_cnt, sparse_tensor.features.shape[1]))
        batch_start = 0
        for batch_idx in range(batch_size):
            n_valid_vox = vox_batch_cnt[batch_idx]
            batch_end = batch_start + n_valid_vox
            out_xyz[batch_idx, :n_valid_vox] = vox_xyz[batch_start: batch_end]
            out_feat[batch_idx, :n_valid_vox] = sparse_tensor.features[batch_start: batch_end]
            # move on
            batch_start = batch_end
        return out_xyz, vox_batch_cnt, out_feat

    def roipool3d_gpu(self, batch_dict):
        """

        Args:
            batch_dict:
                batch_size:
                rois: (B, n_roi, 7 + D)
                multi_scale_3d_features: {'x_conv?': SparseConvTensor} - output of SECOND's backbone
                multi_scale_3d_strides:

        Returns:
            multi_src_pooled_feat (dict):
                x_conv?: (B * n_roi, n_sampled_pts, 3 + C?)
            multi_src_pos_emb (dict):
                x_conv?: (B * n_roi, n_sampled_pts, 27)
            rois_center_corners (torch.Tensor): (B * n_roi, 9, 3)
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']  # (B, n_roi, 7 + D) - in World Frame
        rois_center_corners = self.roi_center_corners(rois, self.model_cfg.ROI_POINT_POOL.MAP_TO_ROI_CANONICAL)
        # rois_center_corners: (B * n_roi, 9, 3)
        n_rois = rois.shape[1]
        n_rois_tot = batch_size * n_rois

        with torch.no_grad():
            multi_src_pooled_feat, multi_src_pos_emb = dict(), dict()
            for src_i, src in enumerate(self.model_cfg.ROI_POINT_POOL.POOL_SRC):
                pts_xyz, pts_feat = self.extract_src_coord_feat(src, batch_dict)
                # pts_xyz: (B, N, 3) - in LiDAR frame
                # pts_feat: (B, N, C)

                pooled_feat, empty_flag = self.roipoint_pool3d_layers[src_i](pts_xyz, pts_feat, rois)
                # pooled_feat: (B, num_rois, num_sampled_points, 3 + C),
                # empty_flag: (B, num_rois)

                pooled_feat = pooled_feat.view(n_rois_tot, -1, pooled_feat.shape[-1])  # (B*n_rois, n_sampled_pts, 3 + C)
                pooled_xyz = pooled_feat[:, :, :3]  # (B*n_rois, n_sampled_pts, 3)
                pooled_feat = pooled_feat[:, :, 3:]  # (B*n_rois, n_sampled_pts, C)

                # map pooled points to ROI Canonical coord
                if self.model_cfg.ROI_POINT_POOL.MAP_TO_ROI_CANONICAL:
                    roi_center = rois[:, :, :3].reshape(-1, 3)  # (B*n_rois, 3)
                    pooled_xyz -= roi_center.unsqueeze(1)
                    pooled_xyz = common_utils.rotate_points_along_z(
                        pooled_xyz, -rois.view(-1, rois.shape[-1])[:, 6]
                    )

                # compute position encoding
                pooled_pos_emb = pooled_xyz.unsqueeze(2) - rois_center_corners.unsqueeze(1)  # (B*n_rois, n_sampled_pts, 9, 3)
                pooled_pos_emb = pooled_pos_emb.reshape(*pooled_pos_emb.shape[:2], -1)  # (B*n_rois, n_sampled_pts, 27)
                test_size(pooled_pos_emb, (n_rois_tot, self.model_cfg.ROI_POINT_POOL.N_POINTS_PER_SRC[src_i], 27), 'pooled_pos_emb')

                # zero-out pooled_xyz & pooled_feat who are empty
                empty_flag = empty_flag.view(-1) > 0
                pooled_xyz[empty_flag] = 0
                pooled_feat[empty_flag] = 0
                pooled_pos_emb[empty_flag] = 0

                # store pooled coord, feat, and position embedding
                multi_src_pooled_feat[src] = torch.cat((pooled_xyz, pooled_feat), dim=-1)
                multi_src_pos_emb[src] = pooled_pos_emb.contiguous()

        return multi_src_pooled_feat, multi_src_pos_emb, rois_center_corners

    @staticmethod
    def roi_center_corners(rois, in_canonical):
        """
        Get rois' center & corners coord in their canonical coord sys
        Args:
            rois (torch.Tensor): (B, N, 7 + D)
            in_canonical (bool): to return center & corners in ROIs' canonical coordinate

        Returns:
            roi_cc: center & corners coordinate (B * N, 9, 3)
        """
        batch_size, n_roi = rois.shape[:2]
        x = torch.tensor([0, 1, 1, 1, 1, -1, -1, -1, -1], device=rois.device).float()
        y = torch.tensor([0, -1, 1, 1, -1, -1, 1, 1, -1], device=rois.device).float()
        z = torch.tensor([0, 1, 1, -1, -1, 1, 1, -1, -1], device=rois.device).float()
        roi_cc = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)  # (9, 3)
        roi_cc = roi_cc.repeat(batch_size, n_roi, 1, 1) * rois[:, :, 3: 6].unsqueeze(2) / 2.0  # (B, N, 9, 3)
        roi_cc = roi_cc.reshape(-1, 9, 3)  # (B * N, 9, 3)
        if in_canonical:
            return roi_cc

        # map roi_cc to world coordinate
        rois_ = rois.view(-1, rois.shape[-1])  # (B * N, 7 + D)
        roi_cc = common_utils.rotate_points_along_z(roi_cc, rois_[:, 6]) + rois_[:, :3].unsqueeze(1)
        assert roi_cc.shape == (batch_size * n_roi, 9, 3)
        return roi_cc

    def forward(self, batch_dict):
        """

        Args:
            batch_dict:

        Returns:

        """
        batch_size = batch_dict['batch_size']
        n_roi = self.model_cfg.TARGET_CONFIG.ROI_PER_IMAGE if self.training else \
            self.model_cfg.NMS_CONFIG.TEST.NMS_POST_MAXSIZE
        tot_n_roi = batch_size * n_roi
        n_sampled_pts = sum(self.model_cfg.ROI_POINT_POOL.N_POINTS_PER_SRC)
        attn_d = self.model_cfg.ATTENTION.DIM

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']  # (B, num_rois)
            assert batch_dict['roi_labels'].shape == (batch_size, n_roi)
            assert torch.min(batch_dict['roi_labels']) > 0

        # =============================================================
        # Pooling
        multi_src_pooled_feat, multi_src_pos_emb, rois_center_corners = self.roipool3d_gpu(batch_dict)

        # match pooled_feat's dim to attention's dim
        multi_src_pooled_coord = dict()
        for i, src in enumerate(self.model_cfg.ROI_POINT_POOL.POOL_SRC):
            multi_src_pooled_coord[src] = multi_src_pooled_feat[src][:, :, :3]
            multi_src_pooled_feat[src] = self.src_dim_matcher[i](multi_src_pooled_feat[src][:, :, 3:]).contiguous()

        # =============================================================
        # Init ROIs' Keys
        rois_key_coord, rois_key_pos_emb, rois_init_feat = self.init_rois_key(rois_center_corners, batch_dict)

        # =============================================================
        # Attention
        # Output:
        #     rois_new_key_feat: (B * n_roi, n_key_feat, attn_d)
        if not self.model_cfg.ATTENTION.SEQUENTIAL:
            # concatenate pooled feat & coord
            pooled_feat, pooled_coord, pooled_pos_emb = list(), list(), list()
            for src in self.model_cfg.ROI_POINT_POOL.POOL_SRC:
                pooled_feat.append(multi_src_pooled_feat[src])
                pooled_coord.append(multi_src_pooled_coord[src])
                pooled_pos_emb.append(multi_src_pos_emb[src])
            # ---
            pooled_feat = torch.cat(pooled_feat, dim=1)  # (B * n_roi, n_sampled_pts, attn_d)
            pooled_coord = torch.cat(pooled_coord, dim=1)  # (B * n_roi, n_sampled_pts, 3)
            pooled_pos_emb = torch.cat(pooled_pos_emb, dim=1)  # (B * n_roi, n_sampled_pts, 27)

            # concatenate ROIs' init feat vector & pos_enc to pooled_feat & pooled_coord before passing to ATTENTION
            if self.mixing_method == 'vector_attn':
                pooled_feat = torch.cat([rois_init_feat, pooled_feat], dim=1).contiguous()
                pooled_coord = torch.cat([rois_key_coord, pooled_coord], dim=1).contiguous()
                test_size(pooled_feat, (tot_n_roi, self.n_key_feat + n_sampled_pts, attn_d), 'pooled_feat')
                test_size(pooled_coord, (tot_n_roi, self.n_key_feat + n_sampled_pts, 3), 'pooled_coord')

            pooled_pos_emb = torch.cat([rois_key_pos_emb, pooled_pos_emb], dim=1).contiguous()
            assert pooled_pos_emb.shape == (tot_n_roi, self.n_key_feat + n_sampled_pts, self.d_init_pos_enc)

            # invoke attention stack to jointly learn ROIs' feat
            rois_new_key_feat = self.attn_stack(pooled_feat, pooled_pos_emb, self.n_key_feat)  # (B * n_roi, n_key_feat, attn_d)
        # ----------------------
        else:
            # use SEQUENTIAL
            rois_new_key_feat = rois_init_feat.contiguous()  # (B * n_roi, n_key_feat, attn_d)
            for i in range(self.model_cfg.ATTENTION.N_REPETITION):
                rois_new_key_feat = self.attn_stack[i](
                    multi_src_pooled_feat,
                    multi_src_pos_emb,
                    self.n_key_feat,
                    src_names=self.model_cfg.ROI_POINT_POOL.POOL_SRC,
                    key_feat=rois_new_key_feat,
                    key_pos_emb=rois_key_pos_emb
                )

        # =============================================================
        # Detection Head
        # concatenate ROIs' key_feat to make ROIs' feat
        rois_new_feat = rois_new_key_feat.reshape(tot_n_roi, -1).unsqueeze(-1).contiguous()  # (B*n_roi, n_key_feat*attn_d, 1)

        # invoke FC (fully connected) Heads to refine ROIs
        shared_feat = self.shared_fc_layer(rois_new_feat)  # (B*n_roi, shared_fc_out, 1)
        test_size(shared_feat, (tot_n_roi, self.model_cfg.SHARED_FC[-1], 1), 'shared_feat')
        rcnn_cls = self.cls_layers(shared_feat).transpose(1, 2).contiguous().squeeze(dim=1)  # (B * n_roi, 1)
        rcnn_reg = self.reg_layers(shared_feat).transpose(1, 2).contiguous().squeeze(dim=1)  # (B * n_roi, 7)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict
        return batch_dict

    def init_rois_key(self, rois_center_corners, batch_dict):
        """
        Create rois' key coord, position embedding & initial feat

        Args:
            rois_center_corners (torch.Tensor): (B * n_roi, 9, 3)
            batch_dict (dict):
                'roi_labels': (B, n_roi)

        Returns:
            rois_key_coord: (B * n_roi, n_key_feat, 3)
            rois_key_pos_emb: (B * n_roi, n_key_feat, 27)
            rois_init_feat: (B * n_roi, n_key_feat, attn_d)
        """
        n_rois_tot = rois_center_corners.shape[0]
        # extract key coord (made of center, corners) from rois_center_corners
        rois_key_coord = rois_center_corners[:, :1]
        test_size(rois_key_coord, (n_rois_tot, self.n_key_feat, 3), 'rois_key_coord')

        # compute position embedding of rois' key
        rois_key_pos_emb = rois_key_coord.unsqueeze(2) - rois_center_corners.unsqueeze(1)
        test_size(rois_key_pos_emb, (n_rois_tot, self.n_key_feat, 9, 3), 'rois_key_pos_emb')
        rois_key_pos_emb = rois_key_pos_emb.reshape(n_rois_tot, self.n_key_feat, -1).contiguous()

        # initialize feat for rois' key
        roi_labels = (batch_dict['roi_labels'] - 1).long()  # (B, n_roi)
        B, n_rois = roi_labels.shape
        attn_d = self.model_cfg.ATTENTION.DIM
        rois_init_feat = []
        for i in range(self.n_key_feat):
            a_key_feat = self.roi_feat_initializer[i](roi_labels)  # (B, n_roi, attn_d)
            test_size(a_key_feat, (B, n_rois, attn_d), 'a_key_feat')
            a_key_feat = a_key_feat.reshape(n_rois_tot, -1).unsqueeze(1)  # (B * n_roi, 1, attn_d)
            rois_init_feat.append(a_key_feat)

        rois_init_feat = torch.cat(rois_init_feat, dim=1).contiguous()  # (B * n_roi, n_key_feat, attn_d)
        test_size(rois_init_feat, (n_rois_tot, self.n_key_feat, attn_d), 'rois_init_feat')

        return rois_key_coord, rois_key_pos_emb, rois_init_feat

    def extract_src_coord_feat(self, src, batch_dict):
        """
        Extract src's xyz & feat from batch_dict

        Args:
            src (str): src name
            batch_dict (dict):
                'multi_scale_3d_features': {'x_conv?': ..., 'x_up?': ...}
                'multi_scale_3d_stride': {'x_conv?': ..., 'x_up?': ...}
                'point_features': output of point-wise backbone
                'point_coords': output of point-wise backbone

        Returns:
            pts_xyz: (B, N, 3) - in LiDAR frame
            pts_feat: (B, N, C)
        """
        batch_size = batch_dict['batch_size']
        if src != 'point_features':
            src_tensor = batch_dict['multi_scale_3d_features'][src]
            src_stride = batch_dict['multi_scale_3d_strides'][src]
            pts_xyz, pts_batch_cnt, pts_feat = self.voxels2pts(
                src_tensor, src_stride, batch_size, output_format='batch'
            )
        else:
            pts_xyz = batch_dict['point_coords']  # (B * N, 4)
            pts_feat = batch_dict['point_features']  # (B * N, C)
            # organize pts_xyz & pts_feat into the from (B, N, C)
            pts_batch_cnt = pts_xyz.new_zeros(batch_size)
            for batch_idx in range(batch_size):
                pts_batch_cnt[batch_idx] = torch.sum(pts_xyz[:, 0] == batch_idx)
            assert pts_batch_cnt.min() == pts_batch_cnt.max(), "Need to have the same number of points across batch"
            pts_xyz = pts_xyz.reshape(batch_size, -1, 4)[:, :, 1:].contiguous()  # (B, N, 3) - in LiDAR frame
            pts_feat = pts_feat.reshape(batch_size, -1, pts_feat.shape[1]).contiguous()  # (B, N, C)

        return pts_xyz, pts_feat


def test_size(tensor: torch.Tensor, expected_size: tuple, name: str):
    assert tensor.shape == expected_size, f"{name}.shape: {tensor.shape}, expect: {expected_size}"

