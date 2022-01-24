from pathlib import Path
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils.calibration_kitti import Calibration
import numpy as np
import os.path as osp
import cv2
from visual_utils.open3d_visual_utils import *
from matplotlib import cm
import argparse
torch.cuda.empty_cache()


def main(cfg_file, ckpt_file, log_file_dir='', split='testing'):
    assert cfg_file != '' and ckpt_file != ''
    log_file = 'log_demo.txt'
    if log_file_dir != '':
        log_file = osp.join(log_file_dir, log_file)
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    cfg_from_yaml_file(cfg_file, cfg)
    cfg.TAG = Path(cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=4,
        logger=None,
        training=False,
        merge_all_iters_to_one_epoch=False,
        total_epochs=10
    )
    train_loader_iter = iter(train_loader)

    # get model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=False)
    model.cuda()
    model.zero_grad()
    model.eval()

    img_dir = f'/home/user/Desktop/libs/OpenPCDet/data/kitti/{split}/image_2'
    calib_dir = f'/home/user/Desktop/libs/OpenPCDet/data/kitti/{split}/calib'

    class_colors_float = np.array([
        [1, 0, 0],  # Car
        [0, 1, 0],  # Pedestrian
        [0, 0, 1]  # Cyclist
    ], dtype=float)  # o3d is RGB
    class_colors = class_colors_float.astype(int) * 255
    class_colors = class_colors[:, ::-1]  # cv2 is BGR
    color_map = cm.rainbow

    # main loop
    batch_dict = None
    for fid in range(20):
        batch_dict = next(train_loader_iter)
        print(f"frame_id: {batch_dict['frame_id']}")

        frame_id = batch_dict['frame_id'][0]
        img = cv2.imread(osp.join(img_dir, f"{frame_id}.png"))
        calib = Calibration(osp.join(calib_dir, f"{frame_id}.txt"))
        point_cloud = batch_dict['points'][:, 1: 4]

        # make prediction
        with torch.no_grad():
            load_data_to_gpu(batch_dict)
            pred_dicts, recall_dicts = model(batch_dict)

        pred_dicts = pred_dicts[0]
        pred_boxes = pred_dicts['pred_boxes'].cpu().numpy()
        pred_scores = pred_dicts['pred_scores'].cpu().numpy()
        pred_lables = pred_dicts['pred_labels'].cpu().numpy()
        pred_accept_mask = pred_scores > 0.35
        # remove boxes behind camera
        front_mask = pred_boxes[:, 0] > 3.0
        pred_accept_mask = np.logical_and(pred_accept_mask, front_mask)
        pred_boxes = pred_boxes[pred_accept_mask]  # (N, 7)
        pred_lables = pred_lables[pred_accept_mask].astype(int) - 1  # (N)

        boxes = pred_boxes
        print(f"boxes: {boxes}")
        if boxes.shape[1] == 8:
            boxes = boxes[:, :7]

        # project boxes to camera
        boxes_ver = box_vertices_from_box_param(boxes)  # (N, 8, 3)
        boxes_ver = boxes_ver.reshape(-1, 3)
        boxes_ver_img, _ = calib.lidar_to_img(boxes_ver)  # (N * 8, 2)

        # ========================
        # drawing
        draw_boxes_on_image(boxes_ver_img.reshape(-1, 8, 2), img, box_colors=class_colors[pred_lables], linewidth=2)
        cv2.imshow(f'cam_left', img)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

        dist = np.linalg.norm(point_cloud, axis=1)
        cloud_colors = color_map(dist / dist.max()) / 2.0
        o3d_pc = create_o3d_point_cloud(point_cloud, colors=cloud_colors[:, :3])
        o3d_boxes = create_o3d_bbox(boxes, colors=class_colors_float[pred_lables])
        o3d.visualization.draw_geometries([o3d_pc, *o3d_boxes], window_name=f'in LiDAR frame {fid}')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='model config file')
    parser.add_argument('--ckpt_file', type=str, default='', help='ckpt to load model weight')
    parser.add_argument('--log_file_dir', type=str, default='', help='directory to save log file')
    args = parser.parse_args()
    main(cfg_file=args.cfg_file, ckpt_file=args.ckpt_file, log_file_dir=args.log_file_dir)

