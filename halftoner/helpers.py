import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import shutil
from IPython import display
import math
import scipy
import glob
import functools
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage as ski


show = False


# ██╗   ██╗██╗███████╗██╗   ██╗ █████╗ ██╗     ██╗███████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
# ██║   ██║██║██╔════╝██║   ██║██╔══██╗██║     ██║╚══███╔╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
# ██║   ██║██║███████╗██║   ██║███████║██║     ██║  ███╔╝ ███████║   ██║   ██║██║   ██║██╔██╗ ██║
# ╚██╗ ██╔╝██║╚════██║██║   ██║██╔══██║██║     ██║ ███╔╝  ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
#  ╚████╔╝ ██║███████║╚██████╔╝██║  ██║███████╗██║███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
#   ╚═══╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                                                                                               

def compare_images_by_channel(imgs, titles=[], figsize=(3, 12), use_gamma=False):
    imgs_no = len(imgs)

    fig_width = figsize[0]*imgs_no
    fig = plt.figure(figsize=(fig_width, figsize[1]))
    fig.subplots_adjust(wspace=0, hspace=0)

    for idx in range(4):
        for img_idx, img in enumerate(imgs):
            ax = fig.add_subplot(4, imgs_no, (imgs_no*idx)+img_idx+1)

            if idx == 3:
                if use_gamma:
                    def gamma(x): return np.power(x, 1/2.2)
                else:
                    def gamma(x): return x

                ax.imshow(gamma(img))
            else:
                ax.imshow(img[:, :, idx])

            if idx == 0 and titles != []:
                ax.set_title(titles[img_idx])

            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()


def display_images_in_grid(imgs, grid_size, titles=[], figsize=(3, 4), gamma=lambda x: x, use_same_max=[]):
    imgs_no = len(imgs)

    fig_width = figsize[0]*grid_size[0]
    fig_height = figsize[1]*grid_size[1]

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.subplots_adjust(wspace=0, hspace=0)

    overall_max = -1

    for img in imgs:
        if np.max(img) > overall_max:
            overall_max = np.max(img)

    for row in range(grid_size[1]):
        for col in range(grid_size[0]):

            img_idx = row*grid_size[0] + col

            ax = fig.add_subplot(grid_size[1], grid_size[0], img_idx+1)

            try:
                if not use_same_max[img_idx]:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)

                    im = ax.imshow(gamma(imgs[img_idx]))
                    fig.colorbar(im, cax=cax, orientation='vertical')
                else:
                    ax.imshow(gamma(imgs[img_idx]), vmin=0, vmax=overall_max)
            except:
                ax.imshow(gamma(imgs[img_idx]))

            if row == 0 and titles != []:
                ax.set_title(titles[col])

            ax.set_xticks([])
            ax.set_yticks([])

    plt.show()


def display_3d_bool_array(arr):
    color_array = np.atleast_3d(np.empty(arr.shape, dtype=object))
    color_array[arr] = '#FFD65D'
    color_array[np.invert(arr)] = '#BFAB6E'

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(np.atleast_3d(np.ones(arr.shape)),
              facecolors=color_array)  # , edgecolors='k')

    plt.show()


def get_array_info(array, name, show=False):
    if show:
        plt.imshow(array)
        plt.title(name)
        plt.colorbar()
        plt.show()\
        
        plt.plot(np.sort(array.flatten()))
        plt.show()


    print('Max: ', np.max(array))
    print('Min: ', np.min(array))
    print('Mean: ', np.mean(array))



def display_image_histograms_rgb(img, title=''):
    data = [np.array(img[:, :, i]).flatten() for i in range(img.shape[-1])]
    plt.hist(data, bins=100, histtype='step', label=['r', 'g', 'b'])
    plt.title(f'{title} histogram')
    plt.show()



# ██╗███╗   ███╗ █████╗  ██████╗ ███████╗                                                         
# ██║████╗ ████║██╔══██╗██╔════╝ ██╔════╝                                                         
# ██║██╔████╔██║███████║██║  ███╗█████╗                                                           
# ██║██║╚██╔╝██║██╔══██║██║   ██║██╔══╝                                                           
# ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗                                                         
# ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝                                                         
                                                                                                
# ███╗   ███╗ █████╗ ███╗   ██╗██╗██████╗ ██╗   ██╗██╗      █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
# ████╗ ████║██╔══██╗████╗  ██║██║██╔══██╗██║   ██║██║     ██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
# ██╔████╔██║███████║██╔██╗ ██║██║██████╔╝██║   ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║╚██╗██║██║██╔═══╝ ██║   ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║██║     ╚██████╔╝███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                                                                                                

def flatten_3d_volume(volume):
    yy, xx, zz = np.atleast_3d(volume).shape

    flat = []

    for z in range(zz):
        for y in range(yy):
            for (x) in range(xx):
                flat.append(volume[y, x, z])
    return flat


# Get the image range of a 2D image per channel
def get_image_range(img):
    img_range = np.zeros(( img.shape[-1], 2), dtype=np.float32)
    for i in range(img.shape[-1]):
        img_range[i, 0] = np.min(img[:, :, i])
        img_range[i, 1] = np.max(img[:, :, i])

    return img_range


# Scale an image to a given range per channel
def change_image_range(img, img_range):
    current_img_range = get_image_range(img)

    scaled_img = np.zeros_like(img)

    for i in range(img.shape[2]):
        scaled_img[:, :, i] = (img[:, :, i] - current_img_range[i, 0]) * \
            (img_range[i, 1] - img_range[i, 0]) / \
            (current_img_range[i, 1] - current_img_range[i, 0]) + img_range[i, 0]

    return scaled_img
    


# ███╗   ███╗██╗███████╗ ██████╗███████╗██╗     ██╗      █████╗ ███╗   ██╗███████╗ ██████╗ ██╗   ██╗███████╗
# ████╗ ████║██║██╔════╝██╔════╝██╔════╝██║     ██║     ██╔══██╗████╗  ██║██╔════╝██╔═══██╗██║   ██║██╔════╝
# ██╔████╔██║██║███████╗██║     █████╗  ██║     ██║     ███████║██╔██╗ ██║█████╗  ██║   ██║██║   ██║███████╗
# ██║╚██╔╝██║██║╚════██║██║     ██╔══╝  ██║     ██║     ██╔══██║██║╚██╗██║██╔══╝  ██║   ██║██║   ██║╚════██║
# ██║ ╚═╝ ██║██║███████║╚██████╗███████╗███████╗███████╗██║  ██║██║ ╚████║███████╗╚██████╔╝╚██████╔╝███████║
# ╚═╝     ╚═╝╚═╝╚══════╝ ╚═════╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝  ╚═════╝ ╚══════╝
                                                                                                          

def get_centred_patch_indices(arr, crop_size):
    yy, xx = arr.shape[0], arr.shape[1]
    crop_height, crop_width = crop_size

    crop_offset_x = xx//2 - crop_width//2
    crop_offset_y = yy//2 - crop_height//2

    return (crop_offset_y, crop_offset_y+crop_height, crop_offset_x, crop_offset_x + crop_width)


# Get a centred patch on an array using get_centred_patch_indices
def get_centred_patch(arr, crop_size):
    if crop_size is None:
        return arr

    crop_offset_y, crop_offset_y_end, crop_offset_x, crop_offset_x_end = get_centred_patch_indices(
        arr, crop_size)
    return arr[crop_offset_y:crop_offset_y_end, crop_offset_x:crop_offset_x_end]


def polyfit_with_fixed_points(n, x, y, xf, yf):
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]

# n, d, f = len(average_albedo), 11, 2

# params = polyfit_with_fixed_points(d, np.linspace(0,len(average_albedo)-1, len(average_albedo)) , sorted_average_albedo
#                                     , [0, len(average_albedo)-1], [sorted_average_albedo[0], sorted_average_albedo[-1]])
# albedo_curve = np.polynomial.Polynomial(params)
# xx = np.linspace(0,len(average_albedo)-1, len(average_albedo))
# plt.plot(np.linspace(0,len(average_albedo)-1, len(average_albedo)), sorted(average_albedo))
# plt.plot([0, len(average_albedo)-1], [sorted_average_albedo[0], sorted_average_albedo[-1]], 'ro')
# plt.plot(xx, albedo_curve(xx), '-')
# plt.show()


def gaussian(x, sigma, mu=0):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))


def get_array_slice_2d(array, crop_size, crop_offset):
    return array[crop_offset[0]:crop_offset[0]+crop_size[0], crop_offset[1]:crop_offset[1]+crop_size[1], :]


def normalize_array(arr, min, max):
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    return (arr - arr_min) / (arr_max - arr_min) * (max - min) + min


def pad_with_border_pattern(array, center_pattern_size, pattern_size):
    additional_dims = len(array.shape) - 2
    additional_axis = [1] * additional_dims

    center_pattern = get_centred_patch(array, center_pattern_size)
    center_pattern_indices = get_centred_patch_indices(
        array, center_pattern_size)

    # plt.imshow(center_pattern)
    # plt.title('center_pattern')
    # plt.show()

    right_pattern_col = center_pattern[:, -pattern_size[0]:]
    left_pattern_col = np.flip(center_pattern[:, :pattern_size[0]], axis=1)

    # plt.imshow(right_pattern_col)
    # plt.title('right_pattern_col')
    # plt.show()

    # plt.imshow(left_pattern_col)
    # plt.title('left_pattern_col')
    # plt.show()

    edge_x_size = (array.shape[1] - center_pattern_size[1]) // 2
    repeats_x = np.ceil(edge_x_size / pattern_size[0])

    right_pattern = np.tile(
        right_pattern_col, (1, int(repeats_x), *additional_axis))
    left_pattern = np.flip(
        np.tile(left_pattern_col, (1, int(repeats_x), *additional_axis)), axis=1)

    # plt.imshow(right_pattern)
    # plt.title('right_pattern')
    # plt.show()

    # plt.imshow(left_pattern)
    # plt.title('left_pattern')
    # plt.show()

    array[center_pattern_indices[0]:center_pattern_indices[1], -
          edge_x_size:] = right_pattern[:, :edge_x_size]
    array[center_pattern_indices[0]:center_pattern_indices[1],
          :edge_x_size] = left_pattern[:, -edge_x_size:]

    # plt.imshow(array)
    # plt.title('array with left and right pattern')
    # plt.show()

    bottom_pattern_row = array[-pattern_size[1] +
                               center_pattern_indices[1]:center_pattern_indices[1], :]
    top_pattern_row = np.flip(
        array[center_pattern_indices[0]:center_pattern_indices[0]+pattern_size[1], :], axis=0)

    # plt.imshow(bottom_pattern_row)
    # plt.title('bottom_pattern_row')
    # plt.show()

    # plt.imshow(top_pattern_row)
    # plt.title('top_pattern_row')
    # plt.show()

    edge_y_size = (array.shape[0] - center_pattern_size[0]) // 2
    repeats_y = np.ceil(edge_y_size / pattern_size[1])

    bottom_pattern = np.tile(
        bottom_pattern_row, (int(repeats_y), 1, *additional_axis))
    top_pattern = np.flip(
        np.tile(top_pattern_row, (int(repeats_y), 1, *additional_axis)), axis=0)

    # plt.imshow(bottom_pattern)
    # plt.title('bottom_pattern')
    # plt.show()

    # plt.imshow(top_pattern)
    # plt.title('top_pattern')
    # plt.show()

    array[-edge_y_size:, :] = bottom_pattern[:edge_y_size, :]
    array[:edge_y_size, :] = top_pattern[-edge_y_size:, :]

    # plt.imshow(array)
    # plt.title('array with all patterns')
    # plt.show()

    return array

# Upsample a given array by a given factor with a variable number of dimensions


def upsample_array(array, factor):
    for i in range(len(factor)):
        array = np.repeat(array, factor[i], axis=i)

    return array


# Return the offset of a random crop of a given size on an array of a given size
def get_random_crop_offset(array_size, crop_size):
    return (np.random.randint(0, array_size[0] - crop_size[0]), 
            np.random.randint(0, array_size[1] - crop_size[1]))

def tile_and_crop( tile, desired_dimensions ):
    tile_size = tile.shape
    repetitions = np.ceil(np.array( [desired_dimensions[0]/tile_size[0], desired_dimensions[1]/tile_size[1]] ) )
    tiled_array = np.tile( tile, (int(repetitions[0]), int(repetitions[1]) ) )
    return tiled_array[:desired_dimensions[0], :desired_dimensions[1]]

