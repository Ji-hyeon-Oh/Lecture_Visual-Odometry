# project1_homography.py

import numpy as np
import cv2
from tqdm import tqdm
import argparse


def choice_random(src_pts, dst_pts):
    random_indices = np.random.choice(len(src_pts), 4)
    return np.array([src_pts[i] for i in random_indices]), np.array([dst_pts[i] for i in random_indices])


def find_homography_ransac(src_pts, dst_pts, good_src_pts, good_dst_pts,
                           threshold=100, iter_limit=2000):

    iter = 0
    best_inlier = 0
    best_H = 0

    while best_inlier < threshold and iter < iter_limit:
        iter += 1
        src, dst = choice_random(good_src_pts, good_dst_pts)
        H = calculate_homography(src, dst)

        inlier = 0
        for j in range(len(src_pts)):
            x = np.transpose(
                np.matrix([src_pts[j][0], src_pts[j][1], 1]))
            u = np.transpose(
                np.matrix([dst_pts[j][0], dst_pts[j][1], 1]))

            # x_hat is estimation result.
            x_hat = np.dot(H, x)
            x_hat = (1/x_hat.item(2))*x_hat

            e = u - x_hat
            d = np.linalg.norm(e)

            if d < 5:
                inlier += 1

            if best_inlier < inlier:
                best_inlier = inlier
                best_H = H

    return best_H


def calculate_homography(src_points, dst_points):
    A = []
    for i in range(len(src_points)):
        x, y = src_points[i][0], src_points[i][1]
        u, v = dst_points[i][0], dst_points[i][1]
        A.append([x, y, 1, 0, 0, 0, -x*u, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

    A = np.array(A)
    _, _, vt = np.linalg.svd(A)

    H = np.reshape(vt[-1], (3, 3))
    H = (1 / H.item(8)) * H
    return H


def create_panorama(image_l, image_r, H):

    # warping
    src_locs = []
    for x in tqdm(range(image_r.shape[1])):
        for y in range(image_l.shape[0]):
            loc = [x, y, 1]
            src_locs.append(loc)
    src_locs = np.array(src_locs).transpose()

    dst_locs = np.matmul(H, src_locs)
    dst_locs = dst_locs / dst_locs[2, :]
    dst_locs = dst_locs[:2, :]
    src_locs = src_locs[:2, :]
    dst_locs = np.round(dst_locs, 0).astype(int)

    height, width, _ = image_l.shape

    # prepare a panorama image
    result = np.zeros((height, width * 2, 3), dtype=int)
    for src, dst in tqdm(zip(src_locs.transpose(), dst_locs.transpose())):
        if dst[0] < 0 or dst[1] < 0 or dst[0] >= width*2 or dst[1] >= height:
            continue
        result[dst[1], dst[0]] = image_r[src[1], src[0]]
    result[0: height, 0: width] = image_l

    print("============finished============")

    return result


def backward_without_interpolation(image, H, output_shape):
    height, width = image.shape[:2]
    warped_image = np.zeros(
        (output_shape[1], output_shape[0], image.shape[2]), dtype=image.dtype)

    H_inv = np.linalg.inv(H)

    for y_out in tqdm(range(output_shape[1])):
        for x_out in range(output_shape[0]):
            point = np.dot(H_inv, np.array([x_out, y_out, 1]))
            point = point / point[2]

            x_in, y_in = int(point[0]), int(point[1])

            if 0 <= x_in < width and 0 <= y_in < height:
                warped_image[y_out, x_out] = image[y_in, x_in]

    print("============finished============")

    return warped_image


def backward_with_interpolation(image, H, output_shape):

    height, width = image.shape[:2]
    warped_image = np.zeros(
        (output_shape[1], output_shape[0], image.shape[2]), dtype=image.dtype)

    H_inv = np.linalg.inv(H)

    for y_out in tqdm(range(output_shape[1])):
        for x_out in range(output_shape[0]):
            # transform the output coordinates to input coordinates using an inverse transformation
            point = np.dot(H_inv, np.array([x_out, y_out, 1]))
            point = point / point[2]

            x_in, y_in = int(point[0]), int(point[1])

            # checks if it is within the input image boundaries
            if 0 <= x_in < width-1 and 0 <= y_in < height-1:
                # bilinear interpolation
                dx, dy = point[0] - x_in, point[1] - y_in
                for channel in range(image.shape[2]):
                    if 0 <= x_in+1 < width and 0 <= y_in+1 < height:
                        warped_image[y_out, x_out, channel] = (
                            (1 - dx) * (1 - dy) * image[y_in, x_in, channel] +
                            dx * (1 - dy) * image[y_in, x_in + 1, channel] +
                            (1 - dx) * dy * image[y_in + 1, x_in, channel] +
                            dx * dy * image[y_in + 1, x_in + 1, channel]
                        )
    print("============finished============")

    return warped_image


def parse_args():
    parser = argparse.ArgumentParser(description="Panorama Creation Script")
    parser.add_argument("--input_left_dir", type=str,
                        required=True, help="Path to the left input image")
    parser.add_argument("--input_right_dir", type=str,
                        required=True, help="Path to the right input image")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Output directory for saving images")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    input_right_dir = args.input_right_dir
    input_left_dir = args.input_left_dir
    output_dir = args.output_dir

    # images load
    image_r = cv2.imread(input_right_dir)  # 오른쪽 사진
    image_l = cv2.imread(input_left_dir)  # 왼쪽 사진

    gray_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
    gray_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)

    # create ORB
    orb = cv2.ORB_create()

    # calculate the keypoints, descriptors
    keypoint_l, descriptor_l = orb.detectAndCompute(gray_l, None)
    keypoint_r, descriptor_r = orb.detectAndCompute(gray_r, None)

    # knnMatch using BF-Hamming
    bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bfmatcher.match(descriptor_l, descriptor_r)

    # sort the result of matching and save good matching
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    good_matches = sorted_matches[:100]

    dst_pts = np.float32(
        [keypoint_l[m.queryIdx].pt for m in matches]).reshape((-1, 2))
    src_pts = np.float32(
        [keypoint_r[m.trainIdx].pt for m in matches]).reshape((-1, 2))

    good_dst_pts = np.float32(
        [keypoint_l[m.queryIdx].pt for m in good_matches]).reshape((-1, 2))
    good_src_pts = np.float32(
        [keypoint_r[m.trainIdx].pt for m in good_matches]).reshape((-1, 2))

    # compute homography matrix
    H = find_homography_ransac(src_pts, dst_pts,
                               good_src_pts, good_dst_pts)

    # result1: forward mapping
    panorama_result = create_panorama(image_l, image_r, H)
    cv2.imwrite(f'{output_dir}result_forward.png', panorama_result)

    # result2: backward mapping (without bilinear interpolation)
    result_bw_wo_ip = backward_without_interpolation(
        image_r, H, (image_r.shape[1] + image_l.shape[1], image_r.shape[0]))
    result_bw_wo_ip[0: image_r.shape[0], 0: image_l.shape[1]] = image_l
    cv2.imwrite(
        f'{output_dir}result_backward_wo_interpolation.png', result_bw_wo_ip)

    # result3: backward mapping with bilinear interpolation
    result_bw_w_ip = backward_with_interpolation(
        image_r, H, (image_r.shape[1] + image_l.shape[1], image_r.shape[0]))
    result_bw_w_ip[0: image_r.shape[0], 0: image_l.shape[1]] = image_l
    cv2.imwrite(
        f'{output_dir}result_backward_w_interpolation.png', result_bw_w_ip)

    # (참고) 위의 코드를 opencv 라이브러리를 이용해서 구현하는 방법
    '''result_usingcv = cv2.warpPerspective(image_r, H, (image_r.shape[1] + image_l.shape[1], image_r.shape[0]))
    result_usingcv[0 : image_r.shape[0], 0 : image_l.shape[1]] = image_l
    cv2.imwrite('result_usingcv.png', result_usingcv)'''
