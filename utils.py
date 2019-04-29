import os
import cv2
import glob
import h5py
import math
import scipy
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.losses import mean_squared_error
from scipy.ndimage.filters import gaussian_filter
from keras.layers import AveragePooling2D
from skimage.measure import compare_psnr, compare_ssim


def get_density_map_gaussian(im, points, adaptive_mode=False, fixed_value=15, fixed_values=None):
    density_map = np.zeros(im.shape[:2], dtype=np.float32)
    h, w = density_map.shape[:2]
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    if adaptive_mode == True:
        fixed_values = None
        leafsize = 2048
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        distances, locations = tree.query(points, k=4)
    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, p[1]), min(w-1, p[0])
        if num_gt > 1:
            if adaptive_mode == 1:
                sigma = int(np.sum(distances[idx][1:4]) * 0.1)
            elif adaptive_mode == 0:
                sigma = fixed_value
        else:
            sigma = fixed_value
        sigma = max(1, sigma)
        gaussian_radius_no_detection = sigma * 3
        gaussian_radius = gaussian_radius_no_detection

        if fixed_values is not None:
            grid_y, grid_x = int(p[0]//(h/3)), int(p[1]//(w/3))
            grid_idx = grid_y * 3 + grid_x
            gaussian_radius = fixed_values[grid_idx] if fixed_values[grid_idx] else gaussian_radius_no_detection
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        gaussian_map[gaussian_map < 0.0003] = 0
        if np.sum(gaussian_map):
            gaussian_map = gaussian_map / np.sum(gaussian_map)
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(density_map.shape[0], p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(density_map.shape[1], p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]
    # density_map[density_map < 0.0003] = 0
    density_map = density_map / (np.sum(density_map / num_gt))
    return density_map

def get_density_map_gaussian_old(im, points, adaptive_mode=0, fixed_value=15, with_direction=False, templates=None, normal_distribution_mask=False):
    density_map = np.zeros(im.shape[:2], dtype=np.float32)
    h, w = density_map.shape[:2]
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    if adaptive_mode == 1:
        # referred from https://github.com/vlad3996/computing-density-maps/blob/master/make_ShanghaiTech.ipynb
        leafsize = 2048
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        distances, locations = tree.query(points, k=4)
    angle_idx = [0, 45, 90, 135]
    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, p[1]), min(w-1, p[0])
        if num_gt > 1:
            if adaptive_mode == 1:
                sigma = int(np.sum(distances[idx][1:4]) * 0.1)
            elif adaptive_mode == 0:
                sigma = fixed_value
        else:
            sigma = fixed_value  # np.average([h, w]) / 2. / 2.
        sigma = max(1, sigma)
        gaussian_radius = sigma * 3

        # filter_mask = np.zeros_like(density_map)
        # gaussian_center = (p[0], p[1])
        # filter_mask[gaussian_center] = 1
        # density_map += gaussian_filter(filter_mask, sigma, mode='constant')

        # If you feel that the scipy api is too slow (gaussian_filter) -- Substitute it with codes below
        # could make it about 100+ times faster, taking around 1.5 minutes on the whole ShanghaiTech dataset A and B.
        if with_direction:
            dt = np.array(distances[idx][1:4]).tolist()
            idx_3 = locations[idx][1:4]
            locations_3 = points[idx_3]
            idx_near = [d for d in range(len(dt)) if dt[d] < gaussian_radius*2]
            distances_3 = distances[idx][1:4][idx_near]
            locations_3 = locations_3[idx_near]
            if len(distances_3) > 1:
                weights_add = []
                for idx_d in range(len(distances_3)):
                    if distances_3[idx_d] == 0:
                        if np.mean(distances_3) == 0:
                            weights_add.append(1/3)
                        else:
                            weights_add.append(1/np.mean(distances_3))
                    else:
                        weights_add.append(1/distances_3[idx_d])
                weights_add = np.array(weights_add) / np.sum(weights_add)
                # print(distances_3, '\n', weights_add)
                angles_3 = []
                for l in locations_3:
                    if l[0] == p[1]:
                        angle = 90
                    elif l[1] == p[0]:
                        angle = 0
                    else:
                        slope = (l[1] - p[0]) / (l[0] - p[1])
                        if np.sin(np.deg2rad(45/2)) < slope < np.sin(np.deg2rad(45/2)):
                            angle = 45
                        elif slope > np.sin(np.deg2rad(90-45/2)) or slope < - np.sin(np.deg2rad(90-45/2)):
                            angle = 90
                        elif - np.sin(np.deg2rad(45/2)) < slope < np.sin(np.deg2rad(45/2)):
                            angle = 0
                        else:
                            angle = 135
                    angles_3.append(angle)
                gaussian_map = np.zeros((gaussian_radius*2+1, gaussian_radius*2+1))
                for ag_idx in range(len(angles_3)):
                    # print(angle_idx.index(angles_3[ag_idx]), gaussian_map.shape, templates[angle_idx.index(angles_3[ag_idx])].shape)
                    temp = cv2.resize(
                        templates[angle_idx.index(angles_3[ag_idx])] * weights_add[ag_idx], (gaussian_radius*2+1, gaussian_radius*2+1),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                    gaussian_map += (temp / np.sum(temp))
            else:
                gaussian_map = np.multiply(
                    cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
                    cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
                )
        else:
            gaussian_map = np.multiply(
                cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
                cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
            )
        if normal_distribution_mask:
            gaussian_map = np.zeros_like(gaussian_map)
            cv2.circle(gaussian_map, (gaussian_radius, gaussian_radius), gaussian_radius//2, 255, -1)
        gaussian_map = gaussian_map / np.sum(gaussian_map)


        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(density_map.shape[0], p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(density_map.shape[1], p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]
    density_map = density_map / (np.sum(density_map / num_gt))
    return density_map


def load_img(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img[:, :, 0]=(img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1]=(img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2]=(img[:, :, 2] - 0.406) / 0.225
    # img[:, :, 0]=(img[:, :, 0] - 0.5) / 1
    # img[:, :, 1]=(img[:, :, 1] - 0.5) / 1
    # img[:, :, 2]=(img[:, :, 2] - 0.5) / 1
    return img.astype(np.float32)


def img_from_h5(path):
    gt_file = h5py.File(path, 'r')
    density_map = np.asarray(gt_file['density'])
    stride = 1
    if stride > 1:
        density_map_stride = np.zeros((np.asarray(density_map.shape).astype(int)//stride).tolist(), dtype=np.float32)
        for r in range(density_map_stride.shape[0]):
            for c in range(density_map_stride.shape[1]):
                density_map_stride[r, c] = np.sum(density_map[r*stride:(r+1)*stride, c*stride:(c+1)*stride])
    else:
        density_map_stride = density_map
    return density_map_stride


def gen_x_y(img_paths, train_val_test='train', augmentation_methods=['ori']):
    x, y = [], []
    for i in img_paths:
        x_ = load_img(i)
        y_ = img_from_h5(i.replace('.jpg', '.h5').replace('images', 'ground'))
        x_, y_ = fix_singular_shape(x_), fix_singular_shape(y_)
        if 'ori' in augmentation_methods:
            x.append(np.expand_dims(x_, axis=0))
            y.append(np.expand_dims(np.expand_dims(y_, axis=0), axis=-1))
        if 'flip' in augmentation_methods and train_val_test == 'train':
            x.append(np.expand_dims(cv2.flip(x_, 1), axis=0))
            y.append(np.expand_dims(np.expand_dims(cv2.flip(y_, 1), axis=0), axis=-1))
    if train_val_test == 'train':
        random_num = random.randint(7, 77)
        random.seed(random_num)
        random.shuffle(x)
        random.seed(random_num)
        random.shuffle(y)
        random.seed(random_num)
        random.shuffle(img_paths)
    return x, y, img_paths


def eval_loss(model, x, y, quality=False):
    preds = []
    for i in x:
        preds.append(np.squeeze(model.predict(i)))
    DM = []
    labels = []
    for i in y:
        DM.append(np.squeeze(i))
        labels.append(round(np.sum(i)))
    losses_DMD = []
    for i in range(len(preds)):
        losses_DMD.append(np.mean(np.square(preds[i] - DM[i]))*5e7)     # mean of Frobenius norm
    loss_DMD = np.mean(losses_DMD)
    losses_MAE = []
    for i in range(len(preds)):
        losses_MAE.append(np.abs(np.sum(preds[i]) - labels[i]))
    losses_MAPE = []
    for i in range(len(preds)):
        losses_MAPE.append(np.abs(np.sum(preds[i]) - labels[i]) / labels[i])
    losses_MSE = []
    for i in range(len(preds)):
        losses_MSE.append(np.square(np.sum(preds[i]) - labels[i]))

    loss_DMD = np.mean(losses_DMD)
    loss_MAE = np.mean(losses_MAE)
    loss_MAPE = np.mean(losses_MAPE)
    loss_MSE = np.sqrt(np.mean(losses_MSE))
    if quality:
        PSNR = []
        SSIM = []
        for i in range(len(preds)):
            data_range = np.max([np.max(preds[i]), np.max(DM[i])])-np.min([np.min(preds[i]), np.min(DM[i])])
            psnr = compare_psnr(preds[i], DM[i], data_range=data_range)
            ssim = compare_ssim(preds[i], DM[i], data_range=data_range)
            PSNR.append(psnr)
            SSIM.append(ssim)
        return loss_DMD, loss_MAE, loss_MAPE, loss_MSE, np.mean(PSNR), np.mean(SSIM)
    return loss_DMD, loss_MAE, loss_MAPE, loss_MSE


def gen_paths(path_file_root='data/paths_train_val_test', dataset='A', with_validation=False):
    path_file_root_curr = os.path.join(path_file_root, 'paths_'+dataset)
    img_paths = []
    paths = os.listdir(path_file_root_curr) if with_validation else os.listdir(path_file_root_curr)[:2]
    for i in sorted([os.path.join(path_file_root_curr, p) for p in paths]):
        with open(i, 'r') as fin:
            img_paths.append([l.rstrip() for l in fin.readlines()])
    return img_paths    # img_paths_test, img_paths_train, img_paths_val


def eval_path_files(dataset="A", validation_split=0.05):
    root = 'data/ShanghaiTech/'
    paths_train = os.path.join(root, 'part_' + dataset, 'train_data', 'images')
    paths_test = os.path.join(root, 'part_' + dataset, 'test_data', 'images')

    img_paths_train = []
    for img_path in glob.glob(os.path.join(paths_train, '*.jpg')):
        img_paths_train.append(str(img_path))
    print("len(img_paths_train) =", len(img_paths_train))
    img_paths_test = []
    for img_path in glob.glob(os.path.join(paths_test, '*.jpg')):
        img_paths_test.append(str(img_path))
    print("len(img_paths_test) =", len(img_paths_test))

    random.shuffle(img_paths_train)
    lst_to_write = [img_paths_train, img_paths_train[:int(len(img_paths_train)*validation_split)], img_paths_test]
    for idx, i in enumerate(['train', 'val', 'test']):
        with open('data/paths_train_val_test/paths_'+dataset+'/paths_'+i+'.txt', 'w') as fout:
            fout.write(str(lst_to_write[idx]))
            print('Writing to data/paths_train_val_test/paths_'+dataset+'/paths_'+i+'.txt')
    return None


def ssim_loss(y_true, y_pred, c1=0.01**2, c2=0.03**2):
    # Generate a 11x11 Gaussian kernel with standard deviation of 1.5
    weights_initial = np.multiply(
        cv2.getGaussianKernel(11, 1.5),
        cv2.getGaussianKernel(11, 1.5).T
    )
    weights_initial = weights_initial.reshape(*weights_initial.shape, 1, 1)
    weights_initial = K.cast(weights_initial, tf.float32)

    mu_F = tf.nn.conv2d(y_pred, weights_initial, [1, 1, 1, 1], padding='SAME')
    mu_Y = tf.nn.conv2d(y_true, weights_initial, [1, 1, 1, 1], padding='SAME')
    mu_F_mu_Y = tf.multiply(mu_F, mu_Y)
    mu_F_squared = tf.multiply(mu_F, mu_F)
    mu_Y_squared = tf.multiply(mu_Y, mu_Y)

    sigma_F_squared = tf.nn.conv2d(tf.multiply(y_pred, y_pred), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_F_squared
    sigma_Y_squared = tf.nn.conv2d(tf.multiply(y_true, y_true), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_Y_squared
    sigma_F_Y = tf.nn.conv2d(tf.multiply(y_pred, y_true), weights_initial, [1, 1, 1, 1], padding='SAME') - mu_F_mu_Y

    ssim = ((2 * mu_F_mu_Y + c1) * (2 * sigma_F_Y + c2)) / ((mu_F_squared + mu_Y_squared + c1) * (sigma_F_squared + sigma_Y_squared + c2))

    return 1 - tf.reduce_mean(ssim, reduction_indices=[1, 2, 3])


def ssim_eucli_loss(y_true, y_pred, alpha=0.001):
    ssim = ssim_loss(y_true, y_pred)
    eucli = mean_squared_error(y_true, y_pred)
    loss = eucli + alpha * ssim
    return loss


def local_sum_loss(y_true, y_pred, alpha=0.5, grid_pooling=3):
    y_true_localized = AveragePooling2D((grid_pooling, grid_pooling))(y_true) * (grid_pooling ** 2)
    y_pred_localized = AveragePooling2D((grid_pooling, grid_pooling))(y_pred) * (grid_pooling ** 2)
    y_true, y_pred = y_true_localized, y_pred_localized
    l1 = K.square(K.mean(K.abs(y_true - y_pred)))
    l2 = K.mean(K.square(y_true - y_pred)) * 1000
    loss = (1-alpha) * l1 + alpha * l2
    loss = loss * 1
    return loss


def random_cropping(x_train, y_train, grid=(2, 2)):
    # Random cropping on training set
    x_train_cropped, y_train_cropped = [], []
    num_crop = grid[0] * grid[1]
    for idx_x in range(len(x_train)):
        wid_patch, hei_patch = int(x_train[idx_x].shape[1] / grid[0]), int(x_train[idx_x].shape[0] / grid[1])
        up_range_x, left_range_x = hei_patch * (grid[0] - 1), wid_patch * (grid[1] - 1)
        # up_range_y, left_range_y = (np.array(y_train[idx_x].shape[0:-1]) * (1 - grid[1])).astype(np.int)
        x_ = x_train[idx_x]
        y_ = y_train[idx_x]
        for _ in range(num_crop):
            up_x = random.randint(0, up_range_x-1)
            left_x = random.randint(0, left_range_x-1)
            x_train_cropped.append(fix_singular_shape(x_[up_x:up_x+hei_patch, left_x:left_x+wid_patch, :]))
            up_y = up_x
            left_y = left_x
            y_train_cropped.append(fix_singular_shape(y_[up_y:up_y+hei_patch, left_y:left_y+wid_patch, :]))
    return np.asarray(x_train_cropped), np.asarray(y_train_cropped)


def fix_singular_shape(tensor):
    # Append 0 lines or colums to fix the shapes as integers times of 8, since there are 3 pooling layers.
    for idx_sp in [0, 1]:
        remainder = tensor.shape[idx_sp] % 8
        if remainder != 0:
            fix_len = 8 - remainder
            pad_list = []
            for idx_pdlst in range(len(tensor.shape)):
                if idx_pdlst != idx_sp:
                    pad_list.append([0, 0])
                else:
                    pad_list.append([int(fix_len/2), fix_len - int(fix_len/2)])
            tensor = np.pad(tensor, pad_list, 'constant')
    return tensor


# def compare_psnr(img1, img2):
#     mse = np.mean(np.square(img1 - img2))
#     if mse:
#         psnr = 10 * np.log10((255**2)/mse)
#     else:
#         psnr = 1e6
#     return psnr
    


# def flip_horizontally(x_train, y_train):
#     # Flip horizontally
#     x_train_flipped, y_train_flipped = [], []
#     for x in x_train:
#         x_train_flipped.append(x[:, :, ::-1, :])
#     for y in y_train:
#         y_train_flipped.append(y[:, :, ::-1, :])
#     x_train += x_train_flipped
#     y_train += y_train_flipped
#     return x_train, y_train


# def data_augmentation(x_train, y_train, augmentation_methods):

#     return x_train, y_train
