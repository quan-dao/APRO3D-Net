import numpy as np
import open3d as o3d
import cv2
from pcdet.utils import common_utils


def create_o3d_point_cloud(pts, colors=None):
    """

        Args:
            pts (np.ndarray): (N, 3) - x, y, z
            colors: (3) or (N, 3) - RGB
        """
    assert pts.shape == (pts.shape[0], 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        if isinstance(colors, tuple) or isinstance(colors, list):
            colors = np.array(colors).reshape(1, -1)
            colors = np.tile(colors, (pts.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def box_vertices_from_box_param(boxes, in_canonical=False):
    """
        Args:
            boxes: (N, 7)

        Returns:
            center_corners: (N, 9, 3) or (N, 8, 3)
        """
    n_box = boxes.shape[0]

    x = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    y = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
    z = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float)

    boxes_cc = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1)  # (8, 3)
    boxes_cc = np.tile(boxes_cc.reshape((1, 8, 3)), (n_box, 1, 1))  # (n_box, 8, 3)
    boxes_cc = boxes_cc * boxes[:, None, 3: 6] / 2.0  # (n_box, 8, 3)
    if in_canonical:
        return boxes_cc

    boxes_cc = common_utils.rotate_points_along_z(boxes_cc, boxes[:, 6]) + boxes[:, :3].reshape(-1, 1, 3)
    return boxes_cc


def create_o3d_bbox(boxes, colors, draw_heading=True):
    """

    Args:
        boxes: (N, 7)
        colors: (3) or (N, 3)
        draw_heading (bool)
    """
    assert boxes.shape == (boxes.shape[0], 7), "Remember to exclude boxes' class to use this function"
    boxes_corners = box_vertices_from_box_param(boxes)  # (N, 8, 3)

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # forward face
        [4, 5], [5, 6], [6, 7], [7, 4],  # back face
        [0, 4], [1, 5], [2, 6], [3, 7],  # horizontal edges
    ]
    if draw_heading:
        lines += [[0, 2], [1, 3]]  # to mark forward face

    if isinstance(colors, list) or isinstance(colors, tuple):
        colors = np.array(colors)
    if colors.shape == (3,):
        colors = np.tile(colors.reshape(1, 3), (boxes.shape[0], 1))

    o3d_boxes = []
    for i in range(boxes.shape[0]):
        bbox = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(boxes_corners[i]),
            lines=o3d.utility.Vector2iVector(lines)
        )
        box_color = np.tile(colors[i].reshape(1, 3), (len(lines), 1))
        bbox.colors = o3d.utility.Vector3dVector(box_color)
        o3d_boxes.append(bbox)
    return o3d_boxes


def draw_boxes_on_image(boxes, image, box_colors=None, linewidth=2) -> None:
    """

    Args:
        boxes (np.ndarray): (N, 8, 2) - 8 for 8 corners; last 2 dim: [pixel_x, pixel_y]
        image (np.ndarray): (H, W, 3)
        box_colors (np.ndarray): (N, 3)
        linewidth (int):
    """

    def draw_rect(img, selected_corners, color, linewidth):
        """ Draw a rectangle (more like a polygon) by connecting 4 points in the image
        Args:
            img (np.ndarray): BGR image, shape (h, w, 3)
            selected_corners (np.ndarray): vertices of rectangle (almost), shape (4, 2)
            color (tuple): B, G, R color
            linewidth (float): width of rectangle edge
        """
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(img,
                     (int(prev[0]), int(prev[1])),
                     (int(corner[0]), int(corner[1])),
                     color, linewidth)
            prev = corner

    assert boxes.shape == (boxes.shape[0], 8, 2), "Project boxes onto image before calling this function"
    for box_idx in range(boxes.shape[0]):
        vertices = boxes[box_idx]
        if box_colors is None:
            colors = (0, 0, 255)
        else:
            colors = tuple(box_colors[box_idx].tolist())
        # Draw the sides
        for i in range(4):
            cv2.line(image,
                     (int(vertices[i, 0]), int(vertices[i, 1])),
                     (int(vertices[i + 4, 0]), int(vertices[i + 4, 1])), colors, linewidth)
        # Draw front (first 4 corners) and rear (last 4 corners)
        draw_rect(image, vertices[:4], colors, linewidth)
        draw_rect(image, vertices[4:], colors, linewidth)

        # draw heading
        cv2.line(image,
                 (int(vertices[0, 0]), int(vertices[0, 1])),
                 (int(vertices[2, 0]), int(vertices[2, 1])), colors, linewidth)
        cv2.line(image,
                 (int(vertices[1, 0]), int(vertices[1, 1])),
                 (int(vertices[3, 0]), int(vertices[3, 1])), colors, linewidth)

