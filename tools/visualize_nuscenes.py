from pyquaternion import Quaternion
import numpy as np
import os.path as osp
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils import splits
import cv2
import json
import open3d as o3d
from nuscenes.utils.geometry_utils import transform_matrix
from pcdet.utils import common_utils
from matplotlib import cm
from visual_utils.open3d_visual_utils import *
import argparse


def main(split, result_file, scene_idx=0, render_cam_back=True, render_point_cloud=True):
    nusc = NuScenes(version=f'v1.0-{split}', dataroot=f'../data/nuscenes/v1.0-{split}', verbose=True)
    if split == 'mini':
        val_scenes_name = splits.mini_val
    elif split == 'test':
        val_scenes_name = splits.test
    else:
        val_scenes_name = splits.val
    val_scenes_token = []
    for scene in nusc.scene:
        if scene['name'] in val_scenes_name:
            val_scenes_token.append(scene['token'])

    scene_token = val_scenes_token[scene_idx]  # something wrong with scene 0, nothing interesting in scene 1 2
    scene = nusc.get('scene', scene_token)

    # read prediction file
    with open(result_file, 'r') as f:
        detections = json.load(f)
    detections = detections['results']

    # display config
    cmap = cm.rainbow
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    class_name_to_idx = dict(zip(class_names, range(len(class_names))))
    class_colors_float = cmap(np.arange(len(class_names)) * 1.0 / len(class_names))[:, :3]
    class_colors = class_colors_float * 255
    class_colors = class_colors.astype(int)

    imsize = (640, 360)
    layout = {
        'CAM_FRONT_LEFT': (0, 0),
        'CAM_FRONT': (imsize[0], 0),
        'CAM_FRONT_RIGHT': (2 * imsize[0], 0),
    }
    horizontal_flip = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    window_name = 'NuScenes'
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)
    if render_cam_back:
        canvas = np.ones((2 * imsize[1], 3 * imsize[0], 3), np.uint8) * 255
        layout_back = {
            'CAM_BACK_LEFT': (0, imsize[1]),
            'CAM_BACK': (imsize[0], imsize[1]),
            'CAM_BACK_RIGHT': (2 * imsize[0], imsize[1]),
        }
        layout.update(layout_back)
    else:
        canvas = np.ones((imsize[1], 3 * imsize[0], 3), np.uint8) * 255

    # =====
    # main loop
    sample_token = scene['first_sample_token']
    frame = 0
    while sample_token != '':
        sample = nusc.get('sample', sample_token)

        # display point cloud & boxes
        # assume: point cloud in LiDAR frame, boxes in global frame
        lidar_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = osp.join(nusc.dataroot, lidar_rec['filename'])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :3]

        # map points in LiDAR to Car
        lidar_cs = nusc.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
        tm_car_from_lidar = transform_matrix(np.array(lidar_cs['translation']), Quaternion(lidar_cs['rotation']))
        points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)  # (N, 4)
        points = tm_car_from_lidar @ points.T  # (4, N)
        points_dist = np.linalg.norm(points[:3, :].T, axis=1)  # (N) to compute color

        # map points in Car to Global
        ego_pos_rec = nusc.get('ego_pose', lidar_rec['ego_pose_token'])
        tm_global_from_car = transform_matrix(np.array(ego_pos_rec['translation']), Quaternion(ego_pos_rec['rotation']))
        points = tm_global_from_car @ points  # (4, N)
        points = points[:3, :].T  # (N, 3)
        print(f"points.shape: {points.shape}")

        # get detection
        dets = detections[sample_token]
        boxes = np.zeros((len(dets), 7))
        scores = np.zeros(len(dets))
        labels = np.zeros(len(dets), dtype=int)
        for idx, det in enumerate(dets):
            boxes[idx, :3] = det['translation']
            boxes[idx, 3] = det['size'][1]  # dx == l
            boxes[idx, 4] = det['size'][0]  # dy == w
            boxes[idx, 5] = det['size'][2]  # dz == h
            rot_mat = Quaternion(det['rotation']).rotation_matrix
            yaw = np.arctan2(rot_mat[1, 0], rot_mat[1, 1])
            boxes[idx, 6] = yaw
            scores[idx] = det['detection_score']
            labels[idx] = class_name_to_idx[det['detection_name']]

        score_thres = 0.5
        boxes = boxes[scores > score_thres]  # (M, 7)
        labels = labels[scores > score_thres]  # (M)
        assert boxes.shape == (boxes.shape[0], 7)
        print(f'boxes.shape: {boxes.shape}')

        boxes_ver = box_vertices_from_box_param(boxes)  # (N, 8, 3)
        boxes_ver = boxes_ver.reshape(-1, 3)  # in global frame

        cam_images = {}
        cam_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        if render_cam_back:
            cam_list += ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for cam_name in cam_list:
            # map to camera
            cam_rec = nusc.get('sample_data', sample['data'][cam_name])
            cam_cs = nusc.get('calibrated_sensor', cam_rec['calibrated_sensor_token'])
            tm_carC_from_cam = transform_matrix(np.array(cam_cs['translation']), Quaternion(cam_cs['rotation']))
            ego_pos_rec = nusc.get('ego_pose', cam_rec['ego_pose_token'])
            tm_global_from_carC = transform_matrix(np.array(ego_pos_rec['translation']),
                                                   Quaternion(ego_pos_rec['rotation']))
            tm_global_from_cam = tm_global_from_carC @ tm_carC_from_cam
            tm_cam_from_global = np.linalg.inv(tm_global_from_cam)
            ver_in_cam = np.concatenate((boxes_ver, np.ones((boxes_ver.shape[0], 1))), axis=1)
            ver_in_cam = tm_cam_from_global @ ver_in_cam.T  # (4, V)
            ver_in_cam = ver_in_cam[:3, :].T  # (M * 8, 3)
            # remove box behind camera
            ver_in_cam = ver_in_cam.reshape(-1, 8, 3)
            behind_mask = np.any(ver_in_cam[:, :, 2] < 1.0, axis=1)  # (M)
            ver_in_cam = ver_in_cam[np.logical_not(behind_mask)]  # (m, 8, 3)
            labels_in_cam = labels[np.logical_not(behind_mask)]
            assert labels_in_cam.shape[0] == ver_in_cam.shape[0]
            if ver_in_cam.shape[0] == 0:
                continue
            assert ver_in_cam.shape == (ver_in_cam.shape[0], 8, 3)
            # project to camera
            ver_in_cam = view_points(ver_in_cam.reshape(-1, 3).T, np.array(cam_cs['camera_intrinsic']), normalize=True)
            ver_in_cam = ver_in_cam[:2, :].T  # (m * 8, 2)
            ver_in_cam = ver_in_cam.reshape(-1, 8, 2)
            # remove boxes have 1 ver outside of image
            img = cv2.imread(osp.join(nusc.dataroot, cam_rec['filename']))
            mask_x = np.logical_and(ver_in_cam[:, :, 0] > 1, ver_in_cam[:, :, 0] < img.shape[1] - 1)
            mask_y = np.logical_and(ver_in_cam[:, :, 1] > 1, ver_in_cam[:, :, 1] < img.shape[0] - 1)
            mask = np.logical_and(mask_x, mask_y)  # (m, 8)
            mask = np.all(mask, axis=1)
            ver_in_cam = ver_in_cam[mask]
            labels_in_cam = labels_in_cam[mask]  # (m)
            colors_in_cam = class_colors[labels_in_cam]
            assert colors_in_cam.shape == (ver_in_cam.shape[0], 3)
            if ver_in_cam.shape[0] > 0:
                draw_boxes_on_image(ver_in_cam, img, colors_in_cam[:, ::-1])

            cam_images[cam_name] = np.copy(img)

        # ============
        # Display
        for channel, img_ in cam_images.items():
            img_ = cv2.resize(img_, imsize)
            if channel in horizontal_flip:
                img_ = img_[:, ::-1, :]
            canvas[
                layout[channel][1]: layout[channel][1] + imsize[1],
                layout[channel][0]: layout[channel][0] + imsize[0], :
            ] = img_

        cv2.imshow(window_name, canvas)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

        if render_point_cloud:
            points_color = cmap(points_dist / points_dist.max()) / 2.0  # (N, 4)
            o3d_points = create_o3d_point_cloud(points, colors=points_color[:, :3])  # colors=points_color[:, :3]
            o3d_boxes = create_o3d_bbox(boxes, colors=class_colors_float[labels])
            o3d.visualization.draw_geometries([o3d_points, *o3d_boxes], window_name=f'Frame: {frame}')

        # move to next sample
        sample_token = sample['next']
        frame += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--split', type=str, default='test', help='split of NuScenes: mini, trainval, test')
    parser.add_argument('--result_file', type=str, default='', help='result_file of the chosen split of NuScenes')
    parser.add_argument('--scene_idx', type=int, default=1, help='index of the scene to be rendered')
    parser.add_argument('--render_cam_back', action='store_true', default=False, help='to show cameras in the back')
    parser.add_argument('--render_point_cloud', action='store_true', default=False, help='to show point cloud')
    args = parser.parse_args()
    main(split=args.split, result_file=args.result_file,
         scene_idx=args.scene_idx, render_cam_back=args.render_cam_back,
         render_point_cloud=args.render_point_cloud)
