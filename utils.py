import os
import cv2
import glob
import h5py
import numpy as np
from random import shuffle
import cv2
import math
import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import keras.backend as K
from keras.losses import mean_squared_error


def get_density_map_gaussian(im, points, adaptive_kernel=False, fixed_value=15):
    density_map = np.zeros(im.shape[:2], dtype=np.float32)
    h, w = density_map.shape[:2]
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    if adaptive_kernel:
        # referred from https://github.com/vlad3996/computing-density-maps/blob/master/make_ShanghaiTech.ipynb
        leafsize = 2048
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        distances = tree.query(points, k=4)[0]

    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, p[1]), min(w-1, p[0])
        if num_gt > 1:
            if adaptive_kernel:
                sigma = int(np.sum(distances[idx][1:4]) * 0.1)
            else:
                sigma = fixed_value
        else:
            sigma = fixed_value  # np.average([h, w]) / 2. / 2.
        sigma = max(1, sigma)

        # filter_mask = np.zeros_like(density_map)
        # gaussian_center = (p[0], p[1])
        # filter_mask[gaussian_center] = 1
        # density_map += gaussian_filter(filter_mask, sigma, mode='constant')

        # If you feel that the scipy api is too slow (gaussian_filter) -- Substitute it with codes below
        # could make it about 100+ times faster, taking around one minute on the whole ShanghaiTech dataset.

        gaussian_radius = sigma * 2
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
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
    return density_map


def load_img(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img[:, :, 0]=(img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1]=(img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2]=(img[:, :, 2] - 0.406) / 0.225
    return img


def img_from_h5(path):
    gt_file = h5py.File(path, 'r')
    density_map = np.asarray(gt_file['density'])
    stride = 8
    density_map_stride = np.zeros((np.asarray(density_map.shape).astype(int)//stride).tolist())
    for r in range(density_map_stride.shape[0]):
        for c in range(density_map_stride.shape[1]):
            density_map_stride[r, c] = np.sum(density_map[r*stride:(r+1)*stride, c*stride:(c+1)*stride])
    return density_map_stride


def gen_x_y(img_paths, train_val_test='train'):
    if train_val_test == 'train':
        shuffle(img_paths)
    x, y = [], []
    for i in img_paths:
        x_ = load_img(i)
        x.append(np.expand_dims(x_, axis=0))
        y_ = img_from_h5(i.replace('.jpg', '.h5').replace('images', 'ground'))
        y.append(np.expand_dims(np.expand_dims(y_, axis=0), axis=-1))
    return x, y, img_paths


def eval_loss(model, x, y):
    preds = []
    for i in x:
        preds.append(np.squeeze(model.predict(i)))
    labels = []
    for i in y:
        labels.append(np.squeeze(i))
    losses_DMD = []
    for i in range(len(preds)):
        losses_DMD.append(np.sum(np.abs(preds[i] - labels[i])))
    loss_DMD = np.mean(losses_DMD)
    losses_MAE = []
    for i in range(len(preds)):
        losses_MAE.append(np.abs(np.sum(preds[i]) - np.sum(labels[i])))
    loss_DMD = np.mean(losses_DMD)
    loss_MAE = np.mean(losses_MAE)
    return loss_DMD, loss_MAE


def gen_paths(path_file_root='data/paths_train_val_test', dataset='A'):
    path_file_root_curr = os.path.join(path_file_root, 'paths_'+dataset)
    img_paths = []
    for i in sorted([os.path.join(path_file_root_curr, p) for p in os.listdir(path_file_root_curr)]):
        with open(i, 'r') as fin:
            img_paths.append(eval(fin.read()))
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

    from random import shuffle
    shuffle(img_paths_train)
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
