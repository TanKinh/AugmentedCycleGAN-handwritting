# Simple augmentation techniques
# =============================================================================
import glob
import inspect
import os
import math
from ast import literal_eval

import cv2
import numpy as np
import numpy.random as random
from imgaug import augmenters as iaa
from imgaug import imgaug as ia
from imgaug.parameters import (Choice, Deterministic, DiscreteUniform, Normal,
                               StochasticParameter, Uniform)
from scipy.ndimage import rotate

from dataloader.utils.misc import read_config, show_image, DEFAULT_CONFIG_PATH
from dataloader.utils.preprocessing import skeletonize_image

# pylint: disable=E1101
ia.CURRENT_RANDOM_STATE = np.random.RandomState(
    np.random.randint(0, 10**6, 1)[0])


# INDIVIDUAL AUGMENTATORS
class TransformPerspective(iaa.PerspectiveTransform):
    """Rewrite the default perspective transform, which has random cropping"""

    def __init__(self, scale=0, cval=255, keep_size=True,
                 name=None, deterministic=False, random_state=None):
        super(TransformPerspective, self).__init__(
            scale=scale, keep_size=keep_size, name=name,
            deterministic=deterministic, random_state=random_state)

        self.cval = cval

    def _create_matrices(self, shapes, random_state):
        """Create the transformation matrix

        # Arguments
            shapes [list of tuples]: list of image shapes
            random_state [numpy Random state]: some random state

        # Returns
            [list of np array]: list of transformation matrices
            [list of ints]: list of heights
            [list of ints]: list of widths
        """
        matrices = []
        max_heights = []
        max_widths = []
        nb_images = len(shapes)
        seeds = ia.copy_random_state(random_state).randint(
            0, 10**6, (nb_images,))

        for _idx in range(nb_images):
            height, width = shapes[_idx][:2]

            pts1 = np.float32([
                [0, 0], [0, height-1], [width-1, 0], [width-1, height-1]
            ])

            transition = self.jitter.draw_samples((4, 2),
                                                  random_state=ia.new_random_state(seeds[_idx]))
            transition[:, 0] = transition[:, 0] * np.min([height, width])
            transition[:, 1] = transition[:, 1] * np.min([height, width])
            transition = transition.astype(np.int32)
            transition[:, 0] = transition[:, 0] + \
                np.abs(np.min(transition[:, 0]))
            transition[:, 1] = transition[:, 1] + \
                np.abs(np.min(transition[:, 1]))

            pts2 = np.float32([
                [transition[0, 0], transition[0, 1]],
                [transition[1, 0], height-1+transition[1, 1]],
                [width-1+transition[2, 0], transition[2, 1]],
                [width-1+transition[3, 0], height-1+transition[3, 1]]
            ])

            height = np.max(pts2[:, 1])
            width = np.max(pts2[:, 0])

            matrices.append(cv2.getPerspectiveTransform(pts1, pts2))
            max_heights.append(height)
            max_widths.append(width)

        return matrices, max_heights, max_widths

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        if not self.keep_size:
            result = list(result)

        matrices, max_heights, max_widths = self._create_matrices(
            [image.shape for image in images],
            random_state
        )

        for i, (M, max_height, max_width) in enumerate(zip(matrices, max_heights, max_widths)):
            # cv2.warpPerspective only supports <=4 channels
            #ia.do_assert(images[i].shape[2] <= 4, "PerspectiveTransform is currently limited to images with 4 or less channels.")
            nb_channels = images[i].shape[2]
            if nb_channels <= 4:
                warped = cv2.warpPerspective(
                    images[i], M, (max_width, max_height), borderValue=self.cval)
                if warped.ndim == 2 and images[i].ndim == 3:
                    warped = np.expand_dims(warped, 2)
            else:
                # warp each channel on its own, re-add channel axis, then stack
                # the result from a list of [H, W, 1] to (H, W, C).
                warped = [
                    cv2.warpPerspective(
                        images[i][..., c], M, (max_width, max_height),
                        borderValue=self.cval)
                    for c in sm.xrange(nb_channels)]
                warped = [warped_i[..., np.newaxis] for warped_i in warped]
                warped = np.dstack(warped)
            #print(np.min(warped), np.max(warped), warped.dtype)
            if self.keep_size:
                h, w = images[i].shape[0:2]
                warped = ia.imresize_single_image(
                    warped, (h, w), interpolation="cubic")
            result[i] = warped

        return result


class ItalicizeLine(iaa.meta.Augmenter):
    """
    Drop-in replace for shear transformation in iaa.Affine (the implementation
    inside iaa.Affine crop images while italicizee)
    """

    def __init__(self, shear=(-40, 41), cval=255, vertical=False,
                 name=None, deterministic=False, random_state=None):
        """Initialize the augmentator

        # Arguments
            shear [float or tuple of 2 floats]: if it is a single number, then
                image will be sheared in that degree. If it is a tuple of 2
                numbers, then the shear value will be chosen randomly
            cval [int]: fill-in value to new pixels
        """
        super(ItalicizeLine, self).__init__(name=name,
                                            deterministic=deterministic, random_state=random_state)

        if isinstance(shear, StochasticParameter):
            self.shear = shear
        elif ia.is_single_number(shear):
            self.shear = Deterministic(shear)
        elif ia.is_iterable(shear):
            ia.do_assert(
                len(shear) == 2,
                "Expected rotate tuple/list with 2 entries, got {} entries."
                .format((len(shear))))
            ia.do_assert(
                all([ia.is_single_number(val) for val in shear]),
                "Expected floats/ints in shear tuple/list.")
            self.shear = Uniform(shear[0], shear[1])
        else:
            raise Exception(
                "Expected float, int, tuple/list with 2 entries or "
                "StochasticParameter. Got {}.".format(type(shear)))

        self.cval = cval
        self.vertical = vertical

    def _augment_images(self, images, random_state, parents, hooks):
        """Augment the images

        # Arguments
            images [list of np array]: the list of images

        # Returns
            [list of np array]: the list of augmented images
        """
        result = images

        seed = random_state.randint(0, 10**6, 1)[0]
        shear_values = self.shear.draw_samples((len(images),),
                                               random_state=ia.new_random_state(seed + 80))

        for _idx, image in enumerate(result):
            angle = shear_values[_idx]
            if angle == 0:
                continue

            if self.vertical:
                # use horizontal italicization method
                image = rotate(image, -90, order=1, cval=self.cval)

            height, original_width, _ = image.shape
            distance = int(height * math.tan(math.radians(math.fabs(angle))))

            if angle > 0:
                point1 = np.array(
                    [[0, 0], [0, height], [5, 0]], dtype=np.float32)
                point2 = np.array(
                    [[distance, 0], [0, height], [5 + distance, 0]],
                    dtype=np.float32)
                image = np.concatenate(
                    [image,
                     np.ones((height, distance, 1), dtype=np.uint8) * self.cval],
                    axis=1)
            else:
                point1 = np.array(
                    [[distance, 0], [distance, height], [distance + 5, 0]],
                    dtype=np.float32)
                point2 = np.array([[0, 0], [distance, height], [5, 0]],
                                  dtype=np.float32)
                image = np.concatenate(
                    [np.ones((height, distance, 1), dtype=np.uint8) * self.cval,
                     image],
                    axis=1)

            height, width, _ = image.shape
            matrix = cv2.getAffineTransform(point1, point2)
            image = cv2.warpAffine(image, matrix, (width, height),
                                   borderValue=self.cval)

            if self.vertical:
                # use horizontal intalicization method
                image = rotate(image, 90, order=1, cval=self.cval)

            if image.ndim == 2:
                image = image[..., np.newaxis]

            result[_idx] = image

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.shear, self.cval]


class RotateLine(iaa.meta.Augmenter):
    """
    Drop-in replace for imgaug's Affine's rotation as the supplied rotation
    does not support fill in cval. With probability 60%
    """

    def __init__(self, angle=(-10, 10), cval=255, name=None,
                 deterministic=False, random_state=None):
        """Initialize the augmentator

        # Arguments
            angle [float or tuple of 2 floats]: if it is a single number, then
                image will be rotated in that degree. If it is a tuple of 2
                numbers, then the angle value will be chosen randomly
            cval [int]: fill-in value to new pixels
        """
        super(RotateLine, self).__init__(name=name,
                                         deterministic=deterministic, random_state=random_state)

        if isinstance(angle, StochasticParameter):
            self.angle = angle
        elif ia.is_single_number(angle):
            self.angle = Deterministic(angle)
        elif ia.is_iterable(angle):
            ia.do_assert(
                len(angle) == 2,
                "Expected rotate tuple/list with 2 entries, got {} entries."
                .format((len(angle))))
            ia.do_assert(
                all([ia.is_single_number(val) for val in angle]),
                "Expected floats/ints in angle tuple/list.")
            self.angle = Uniform(angle[0], angle[1])
        else:
            raise Exception(
                "Expected float, int, tuple/list with 2 entries or "
                "StochasticParameter. Got {}.".format(type(angle)))

        self.cval = cval

    def _augment_images(self, images, random_state, parents, hooks):
        """Augment the images

        # Arguments
            images [list of np array]: the list of images

        # Returns
            [list of np array]: the list of augmented images
        """
        result = images

        seed = random_state.randint(0, 10**6, 1)[0]
        angle_values = self.angle.draw_samples((len(images),),
                                               random_state=ia.new_random_state(seed + 90))

        for _idx, image in enumerate(result):
            angle = angle_values[_idx]
            if angle == 0:
                continue
            result[_idx] = rotate(image, angle, order=1, cval=self.cval)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.angle, self.cval]


class BackgroundImageNoises(iaa.meta.Augmenter):
    """
    Add background noises into an image
    """

    def __init__(self, background_folder, name=None,
                 deterministic=False, random_state=None):
        """Initialize the background noise image addition"""

        super(BackgroundImageNoises, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.background_folder = os.path.expanduser(background_folder)
        self.background_noise_files = []
        self.background_noise_files += glob.glob(
            os.path.join(self.background_folder, '**', '*.png'), recursive=True)
        self.background_noise_files += glob.glob(
            os.path.join(self.background_folder, '**', '*.jpg'), recursive=True)
        self.background_noise_files += glob.glob(
            os.path.join(self.background_folder, '**', '*.PNG'), recursive=True)
        self.background_noise_files += glob.glob(
            os.path.join(self.background_folder, '**', '*.JPG'), recursive=True)

    def _augment_images(self, images, random_state, parents, hooks):
        """Augment the images

        # Arguments
            images [list of np array]: the list of images

        # Returns
            [list of np array]: the list of augmented images
        """
        result = images
        for _idx, each_image in enumerate(images):
            background = cv2.imread(
                random.choice(self.background_noise_files),
                cv2.IMREAD_GRAYSCALE)
            height, width = each_image.shape[:2]
            background = cv2.resize(background, (width, height),
                                    interpolation=cv2.INTER_NEAREST)

            image = cv2.bitwise_and(each_image, background)
            if image.ndim == 2:
                image = image[..., np.newaxis]

            result[_idx] = image

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.background_folder]


class PencilStroke(iaa.meta.Augmenter):
    """
    Transform the image to have pencil stroke
    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        """Initialize the augmentator"""
        super(PencilStroke, self).__init__()

    def _augment_images(self, images, random_state, parents, hooks):
        """Construct pencil stroke

        This method works by binarize an image, and then transform the stroke
        to have a Gaussian distribution around some mean and with certain
        distribution.

        # Arguments
            image [np array]: the character image that we will transform
                stroke to pencil stroke. This image is expected to have dark
                strokes on white background

        # Returns
            [np array]: the line text image with pencil stroke
        """
        result = images

        for _idx, each_image in enumerate(images):
            image = each_image[:, :, 0]

            # mean pixel in range 140 - 170, distribution in range 0.08 - 0.2
            mean_pixel = random.choice(30) + 140
            distribution = (random.choice(12) + 8) / 100

            # Binarize and invert the image to have foreground 1 & background 0
            bin_image = cv2.threshold(
                image, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            reconstruct_image = (
                255 - (bin_image * (255 - mean_pixel))).astype(np.float32)

            # Create a Gaussian kernel
            kernel_scale = random.choice(2) + 2
            small_shape = (int(image.shape[0]/kernel_scale),
                           int(image.shape[1]/kernel_scale))
            large_shape = (image.shape[1], image.shape[0])  # width x height
            kernel = cv2.resize(
                random.normal(0, distribution, small_shape),
                large_shape,
                interpolation=cv2.INTER_NEAREST).astype(np.float32)

            image = reconstruct_image + reconstruct_image * bin_image * kernel
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = cv2.blur(image, (2, 2))
            if random.random() < 0.3:
                image = cv2.blur(image, (2, 2))

            image = image[..., np.newaxis]
            result[_idx] = image

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return []


class Skeletonize(iaa.meta.Augmenter):
    """
    Randomly skeletonize the image
    """

    def __init__(self, is_binary, name=None, deterministic=False,
                 random_state=None):
        """Initialize the augmentator"""
        super(Skeletonize, self).__init__()

        self.is_binary = is_binary

    def _augment_images(self, images, random_state, parents, hooks):
        """Augment the images

        # Arguments
            images [list of np array]: the list of images

        # Returns
            [list of np array]: the list of augmented images
        """
        result = images

        for _idx, each_image in enumerate(images):
            image = each_image[:, :, 0]

            dilation_value = 5 if random.random() <= 0.5 else 3
            image = skeletonize_image(image, dilation_value, self.is_binary)

            image = image[..., np.newaxis]
            result[_idx] = image

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.is_binary]


# SEQUENTIAL AUGMENT OBJECT
class Augment(object):
    """Augmentation object"""

    def __init__(self, config, is_binary=False):
        """Initialize all augmentators"""

        # Meta-information
        self.config = config
        self.config_short = None
        self.is_binary = is_binary

        if 'cval' in config.keys() and config['cval']:
            self.cval = self.config['cval']
        else:
            self.cval = 1 if self.is_binary else 255
        self.random_state = self.config['random_state'] if 'random_state' in self.config.keys(
        ) else None
        self.grayscale_only = 0 if self.is_binary else 1
        # Initial value
        self.background_images_path = '.'
        self.background_images_value = 0

        # Augmentation scheme applies to each character
        self.char_piecewise_affine_1 = iaa.PiecewiseAffine(
            scale=0.01, nb_cols=15, nb_rows=10, cval=self.cval)
        self.char_piecewise_affine_2 = iaa.PiecewiseAffine(
            scale=0.01, nb_cols=7, nb_rows=5, cval=self.cval)
        self.augment_lines_general = iaa.Sequential([])
        self.augment_lines_short_image = iaa.Sequential([])

        # Augmentatin pipeline for the whole image
        if not self.config.get('augmentation_pipeline', None):
            # Use mild augmentation config if augmentation_pipeline not existed or it's empty or None
            if self.config['image_type'] == 'binary':
                if self.config['scheme'] == 'printed':
                    self.config = read_config(os.path.join(
                        DEFAULT_CONFIG_PATH, 'printed_mild_binary.yaml'))
                elif self.config['scheme'] == 'handwriting':
                    self.config = read_config(os.path.join(
                        DEFAULT_CONFIG_PATH, 'handwriting_mild_binary.yaml'))
            elif self.config['image_type'] == 'grayscale':
                if self.config['scheme'] == 'printed':
                    self.config = read_config(
                        os.path.join(DEFAULT_CONFIG_PATH, 'printed_mild_gray.yaml'))
                elif self.config['scheme'] == 'handwriting':
                    self.config = read_config(
                        os.path.join(DEFAULT_CONFIG_PATH, 'handwriting_mild_gray.yaml'))

        # Load default short augmentation pipeline
        if self.config['image_type'] == 'binary':
            if self.config['scheme'] == 'printed':
                self.config_short = read_config(
                    os.path.join(DEFAULT_CONFIG_PATH, 'printed_mild_short_binary.yaml'))
            elif self.config['scheme'] == 'handwriting':
                self.config_short = read_config(
                    os.path.join(DEFAULT_CONFIG_PATH, 'handwriting_mild_short_binary.yaml'))

        elif self.config['image_type'] == 'grayscale':
            if self.config['scheme'] == 'printed':
                self.config_short = read_config(
                    os.path.join(DEFAULT_CONFIG_PATH, 'printed_mild_short_gray.yaml'))
            elif self.config['scheme'] == 'handwriting':
                self.config_short = read_config(
                    os.path.join(DEFAULT_CONFIG_PATH, 'handwriting_mild_short_gray.yaml'))

    def add_background_image_noises(self, folder_path):
        """Set the folder path and enable add background images

        # Arguments
            folder_path [str]: the path to folder containing noise images
        """
        self.background_images_path = folder_path
        self.background_images_value = 1

    def reseed(self):
        """Reseed all the augmentators"""
        self.char_piecewise_affine_1.reseed()
        self.char_piecewise_affine_2.reseed()
        self.augment_lines_general.reseed()

    def augment_line(self, image):
        """Augment a text line image

        This will make use of `augment_lines_short_image`

        # Arguments
            image [np array]: the image to augment

        # Returns
            [np array]: the augmented image
        """
        if image.shape[1] / image.shape[0] < 2.5:
            #w/h < 2.5 as short image
            return self.augment_lines_short_image.augment_image(image)

        return self.augment_lines_general.augment_image(image)

    def build_augmentators(self):
        """The augmentators that will be used"""
        total_pipeline = process_list_augment(
            self.config['augmentation_pipeline'], cval=self.cval, random_state=self.random_state, is_binary=self.is_binary)
        self.augment_lines_general = iaa.Sequential(
            total_pipeline, random_order=False)
        total_short_pipeline = process_list_augment(
            self.config_short['augmentation_pipeline'], cval=self.cval, random_state=self.random_state, is_binary=self.is_binary)
        self.augment_lines_short_image = iaa.Sequential(
            total_short_pipeline, random_order=False)


class HandwritingCharacterAugment(Augment):
    """Augmentation scheme for character"""

    def __init__(self, is_binary=False, cval=None):
        """Initialize the augmentator object"""
        super(HandwritingCharacterAugment, self).__init__(
            is_binary=is_binary, cval=cval)
        self.augment_character = None

    def reseed(self):
        """Reseed"""
        super(HandwritingCharacterAugment, self).reseed()
        self.augment_character.reseed()

    def build_augmentators(self):
        """Build the augmentator"""
        self.augment_character = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.PiecewiseAffine(
                    scale=0.01, nb_cols=15, nb_rows=10, cval=self.cval),
                iaa.PiecewiseAffine(
                    scale=0.01, nb_cols=7, nb_rows=5, cval=self.cval)
            ])),
            iaa.Sometimes(0.3, ItalicizeLine(shear=(-20, 21), cval=self.cval)),
            iaa.Sometimes(0.3, ItalicizeLine(shear=(-20, 21), vertical=True,
                                             cval=self.cval)),
            iaa.Sometimes(0.3, RotateLine(angle=(-5, 5), cval=self.cval)),
            iaa.Sometimes(0.3,
                          PerspectiveTransform(
                              (0.05, 0.15), cval=self.cval, keep_size=False)),
            iaa.Sometimes(0.3, iaa.ElasticTransformation(
                alpha=(0, 1.0), sigma=(0.4, 0.6), cval=self.cval)),
            iaa.Sometimes(0.02, Skeletonize(self.is_binary)),
            iaa.Sometimes(0.1 * self.grayscale_only,
                          iaa.ContrastNormalization((0.5, 1.5))),
            iaa.Sometimes(0.3 * self.grayscale_only, PencilStroke()),
            iaa.Sometimes(
                0.3 * self.grayscale_only * self.background_images_value,
                BackgroundImageNoises(self.background_images_path)),
            iaa.Sometimes(self.grayscale_only, iaa.OneOf([
                iaa.Sometimes(0.5, iaa.OneOf([
                    iaa.GaussianBlur((0.2, 1.0)),
                    iaa.AverageBlur(k=(1, 5)),
                    iaa.MedianBlur(k=(1, 3))
                ])),
                iaa.OneOf([
                    iaa.Add((-50, 30)),
                    iaa.Multiply((0.9, 1.1)),
                    iaa.OneOf([
                        iaa.Dropout(p=(0.01, 0.05)),
                        iaa.CoarseDropout(
                            (0.01, 0.02), size_percent=(0.1, 0.25))
                    ]),
                    iaa.Sometimes(0.7, iaa.OneOf(
                        [iaa.AddElementwise((-10*n, 5*n)) for n in range(1, 5)]
                        + [iaa.AdditiveGaussianNoise(scale=(0.05*255, 0.1*255)),
                           iaa.MultiplyElementwise((0.95, 1.05))]
                    ))
                ])
            ]))
        ])


def process_list_augment(input_list, cval, random_state, is_binary):
    """Convert dictionary augmentation configs into augmentation object"""
    current_pipeline = []
    for _, each in enumerate(input_list):
        function_name = next(iter(each))
        augment_param = each[function_name]
        if isinstance(augment_param, list):
            prob = 1
            for each_element in augment_param:
                if isinstance(each_element, dict) and 'prob' in each_element.keys():
                    prob = each_element['prob']
                    augment_param.remove(each_element)
                    break
            big_object = None
            if function_name == 'Sequential':
                for each_element in augment_param:
                    if isinstance(each_element, dict) and 'random_order' in each_element.keys():
                        random_order = each_element['random_order']
                        augment_param.remove(each_element)
                        big_object = iaa.Sequential(process_list_augment(
                            input_list=augment_param, cval=cval,
                            random_state=random_state, is_binary=is_binary), random_order=random_order, random_state=random_state)
            elif function_name == 'SomeOf':
                for each_element in augment_param:
                    if isinstance(each_element, dict) and 'n' in each_element.keys():
                        n = each_element['n']
                        augment_param.remove(each_element)
                        big_object = iaa.SomeOf(literal_eval(n), process_list_augment(
                            input_list=augment_param, cval=cval, random_state=random_state, is_binary=is_binary), random_state=random_state)
            elif function_name == 'OneOf':
                big_object = iaa.OneOf(process_list_augment(
                    input_list=augment_param, cval=cval, random_state=random_state, is_binary=is_binary), random_state=random_state)
            if prob:
                big_object = iaa.Sometimes(
                    prob, big_object, random_state=random_state)
            current_pipeline.append(big_object)
        elif isinstance(augment_param, dict):
            prob = augment_param.pop('prob', None)
            for key, value in augment_param.items():
                if isinstance(value, str) and value[0] == '(' and value[-1] == ')' and len(value) > 2:
                    augment_param[key] = literal_eval(value)
            try:
                augment_function = getattr(iaa, function_name)
            except AttributeError:
                print(function_name, 'not found in imgaug lib, using custom')
                augment_function = globals()[function_name]
            required_param = list(inspect.getargspec(augment_function))[0]
            if 'cval' in required_param:
                augment_param['cval'] = cval
            if 'is_binary' in required_param:
                augment_param['is_binary'] = is_binary
            augment_param['random_state'] = random_state
            augment_object = augment_function(**augment_param)
            if prob is not None:
                augment_object = iaa.Sometimes(
                    prob, augment_object, random_state=random_state)
            current_pipeline.append(augment_object)
    return current_pipeline