import numpy as np
import cv2
import base64

IMG_SIZE = 150  # img resolution -> IMG_SIZE x IMG_SIZE


def read_b64(encoded_data):
    np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def auto_pad(img):
    assert len(img.shape) == 3
    s = max(img.shape[0:2])
    # Creating a dark square with NUMPY
    f = np.zeros((s, s, 3), np.uint8)
    # Getting the centering position
    ax, ay = (s - img.shape[1]) // 2, (s - img.shape[0]) // 2
    # Pasting the 'image' in a centering position
    f[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img
    return f


def is_black_img(img):
    if np.percentile(img, 95) < 50:
        return True
    elif np.mean(img) < 10:
        return True
    else:
        return False


def process_fundus_retinal_1(b64_img):
    img_array = read_b64(b64_img)
    return np.array(cv2.resize(np.array(img_array), (IMG_SIZE, IMG_SIZE)).reshape((1, IMG_SIZE, IMG_SIZE, 3))) / 255


def process_fundus_retinals_1(b64_imgs):
    img_arrays = [read_b64(b64_img) for b64_img in b64_imgs]
    return np.array([np.array(cv2.resize(np.array(img_array), (IMG_SIZE, IMG_SIZE)).reshape((IMG_SIZE, IMG_SIZE, 3))) / 255 for img_array in img_arrays])


def process_fundus_retinal_2(b64_img):
    img_array = read_b64(b64_img)
    if not is_black_img(img_array):
        cropped_array = auto_pad(img_array)
        new_array = cv2.resize(cropped_array, (100, 100)).reshape(1, 100, 100, 3)
        return new_array
    else:
        return BrokenPipeError('image too dark to classify')


def process_fundus_retinals_2(b64_imgs):
    arrays = []
    img_arrays = [read_b64(b64_img) for b64_img in b64_imgs]
    for img_array in img_arrays:
        if not is_black_img(img_array):
            cropped_array = auto_pad(img_array)
            new_array = cv2.resize(cropped_array, (100, 100)).reshape(100, 100, 3)
            arrays.append(new_array)
        else:
            arrays.append(np.zeros((100,100,3), float).reshape(100, 100, 3))
    return np.array(arrays)
