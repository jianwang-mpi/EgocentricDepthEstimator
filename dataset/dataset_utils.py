import numpy as np

import cv2


def generate_target(joints, image_size=256, heatmap_size=64, num_joints=15, sigma=1):
    '''
    :param joints:  [num_joints, 3]
    :param joints_vis: [num_joints, 3]
    :return: target, target_weight(1: visible, 0: invisible)
    '''
    # sigma = 1
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    # target_weight[:, 0] = joints[:, 0]

    target = np.zeros((num_joints,
                       heatmap_size,
                       heatmap_size),
                      dtype=np.float32)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        feat_stride = (image_size / heatmap_size, image_size / heatmap_size)
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size or ul[1] >= heatmap_size \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size)
        img_y = max(0, ul[1]), min(br[1], heatmap_size)

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target


def get_bbox_from_joints(joints_with_confidence, img_scale=200., vis_thr=0.2):
    # kps = joints_with_confidence
    # Pick the most confident detection.
    # scores = [np.mean(kp[kp[:, 2] > vis_thr, 2]) for kp in kps]
    # kp = kps[np.argmax(scores)]

    kp = joints_with_confidence
    vis = kp[:, 2] > vis_thr
    vis_kp = kp[vis, :2]
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        print('bad!')
        import ipdb
        ipdb.set_trace()
    center = (min_pt + max_pt) / 2.
    scale = img_scale / person_height

    return scale, center

def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor

def scale_and_crop(image, scale, center, img_size):
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    return crop, proc_param

def process_2D_pose(raw_pose):
    if raw_pose is None:
        return None
    else:
        pose = []  # pose in coco model
        for i in range(0, len(raw_pose), 3):
            x = raw_pose[i]
            y = raw_pose[i + 1]
            confidence = raw_pose[i + 2]
            pose.append(np.asarray((x, y, confidence)))
        neck = pose[1] + (pose[0] - pose[1]) * 0.25
        # neck = pose[1]
        pose_egopose_model = [neck, pose[2], pose[3], pose[4], pose[5], pose[6], pose[7], pose[9], pose[10], pose[11],
                              pose[22], pose[12], pose[13], pose[14], pose[19]]
    return np.asarray(pose_egopose_model), np.asarray(pose)
# add
