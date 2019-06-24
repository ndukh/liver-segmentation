import os

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from medpy.io import load as medpy_load
from PIL import Image as pil_image
from keras_preprocessing.image.utils import _PIL_INTERPOLATION_METHODS


def get_filepaths(folder_path):
    """
    Retrieving of full filenames of dataset content.

    Parameters:
    -----------
    num : batch number (there are two batches of dataset)
    Returns:
    --------
    X_filenames, dataframe of paths
    """
    X_filenames = []
    # for directory in os.listdir(folder_path):
    for dirName, subdirList, fileList in os.walk(folder_path):
        for filename in fileList:
            if ".dcm" in filename.lower():
                X_filenames.append(os.path.join(dirName, filename).replace('\\', '/'))
    return pd.DataFrame(sorted(X_filenames))


# override the original keras load_image method to add DICOM support
def load_img_dicom(path, grayscale=False, color_mode='rgb', target_size=None,
                   interpolation='nearest'):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rbg", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    # to support dicom format
    dicom, _ = medpy_load(path)
    img = pil_image.fromarray(dicom.squeeze())
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rbg", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def get_data_generator(filepaths_df):
    import keras_preprocessing
    keras_preprocessing.image.iterator.load_img = load_img_dicom  # adding dicom support
    gen_params = {
        'x_col': 0,
        'target_size': (512, 512),
        'color_mode': 'grayscale',
        'batch_size': 1,
        'class_mode': None,
        'shuffle': False,
        'validate_filenames': False,
    }
    data_preprocessor = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        data_format='channels_first'
    )
    data_generator = data_preprocessor.flow_from_dataframe(filepaths_df, **gen_params)
    return data_generator


def save_img(img, path):
    img = pil_image.fromarray(img * 255)
    img.save(path)
