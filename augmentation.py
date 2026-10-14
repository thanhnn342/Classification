import cv2 as cv 
import numpy as np 

def divide_img_w_overlap(image, dimension, overlap_ratio):
    '''
    Divide the original image into smaller square patches with overlap 
    '''
    # Get the dimensions of the image

    shape = image.shape

    if len(shape) == 3:
        height, width, _ = image.shape
    
    else:
        height, width = image.shape

    num_pixel_overlap = int(overlap_ratio * dimension) # Calculate number of pixel overlap in each images

    height_parts = (height - num_pixel_overlap) // (dimension - num_pixel_overlap)
    width_parts = (width - num_pixel_overlap) // (dimension - num_pixel_overlap)

    new_height = height_parts * dimension - (height_parts - 1) * num_pixel_overlap
    new_width = width_parts * dimension - (width_parts - 1) * num_pixel_overlap

    img_res = cv.resize(image, dsize = (new_width, new_height))

    (part_height, part_width) = (dimension, dimension)

    # Divide the image into parts

    parts = []
    for i in range(height_parts):
        for j in range(width_parts):
            
            x_start = j * (part_width - num_pixel_overlap)
            x_end   = x_start + part_width

            y_start = i * (part_height - num_pixel_overlap)
            y_end   = y_start + part_height

            part = img_res[y_start : y_end, x_start : x_end]

            parts.append(part)

    return parts, (img_res.shape[0], img_res.shape[1])


def rotate_img(img, angle):
    '''
    Rotate image into defined degree
    '''
    center = (img.shape[1]//2, img.shape[0]//2)
    scale = 1

    rot_mat = cv.getRotationMatrix2D( center, angle, scale )
    rotated_img = cv.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    return rotated_img


def blur(image):
    '''
    Blur image using Gaussian Distribution
    '''
    blured_img = cv.GaussianBlur(image, (21,21), 5)
    return blured_img


def add_salt_and_pepper_noise(image, noise_ratio=0.02):
    '''
    Add noise into image using ratio
    '''
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    noisy_pixels = int(h * w * noise_ratio)
 
    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[row, col] = [0, 0, 0] 
        else:
            noisy_image[row, col] = [255, 255, 255]
 
    return noisy_image


def random_shift(image, shift_range = 10):
    '''
    Randomly shift the image in x and y direction with defined range
    '''
    x_shift = np.random.randint(0, shift_range)
    y_shift = np.random.randint(0, shift_range)
    shift_mat = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shift_img = cv.warpAffine(image, shift_mat, ((image.shape[1], image.shape[0])))

    return shift_img
