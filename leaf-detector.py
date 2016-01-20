# The MIT License (MIT)
#
# Copyright (c) 2016 Olya Matveeva <olya.matveeva@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import warnings

import cv2
from imutils import paths
from localbinarypatterns.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC

from cli import parse_cli
from logger import logger
import settings
from utils import ResisedImage
from utils import SegmentedImage

from tqdm import tqdm


warnings.filterwarnings("ignore")


def en_ru(en):
    if settings.enable_rus and en in settings.dictionary:
        return settings.dictionary[en].encode('utf8')
    else:
        return en


def image_holder(images_dictionary):
    images = [(path, label)
              for label, img in images_dictionary.items()
              for path in reversed(img)]
    while len(images) > 0:
        yield images.pop()


def get_training_images(training_dir):
    training_images = dict()
    for image_path in paths.list_images(training_dir):
                leaf_sort = image_path.split("/")[-2]
                if leaf_sort in training_images:
                    training_images[leaf_sort].append(image_path)
                else:
                    training_images[leaf_sort] = [image_path]
    return training_images


def perform_training(patters_operator, training_images,
                     resize=False, thresholding=False):
    model = LinearSVC(C=100.0, random_state=42)

    data = []
    labels = []

    for imagePath, label in tqdm([i for i in image_holder(training_images)],
                                 leave=True):

        image = cv2.imread(imagePath)

        if resize:
            image = ResisedImage(image, settings.default_size)

        gray_image = cv2.cvtColor(image.resized_image
                                  if hasattr(image, 'resized_image')
                                  else image,
                                  cv2.COLOR_BGR2GRAY)

        if thresholding:
            gray_image = SegmentedImage(gray_image)

        histogram = patters_operator.describe(
            gray_image.segmented_image
            if hasattr(gray_image, 'segmented_image') else gray_image)

        labels.append(label)
        data.append(histogram)

    print '\n'
    model.fit(data, labels)
    return model


def perform_recognizing(patters_operator, model, leaf_images, resize=False,
                        thresholding=False):
    for imagePath in leaf_images:
        image = cv2.imread(imagePath)
        if resize:
            image = ResisedImage(image, settings.default_size)

        gray_image = cv2.cvtColor(image.resized_image
                                  if hasattr(image, 'resized_image')
                                  else image,
                                  cv2.COLOR_BGR2GRAY)
        if thresholding:
            gray_image = SegmentedImage(gray_image)

        histogram = patters_operator.describe(
            gray_image.segmented_image
            if hasattr(gray_image, 'segmented_image') else gray_image)
        prediction = model.predict(histogram)[0]

        logger.debug('Image [{0}] is recognized as "{1}" leaf'.format(
            imagePath, en_ru(prediction)))

        # display the image and the prediction
        cv2.putText(image.original_image
                    if hasattr(image, 'original_image') else image,
                    prediction,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow("Image",
                   image.original_image
                   if hasattr(image, 'original_image') else image)
        cv2.waitKey(0)


def main():

    leaf_images, training_dir, thresholding, resize = parse_cli()

    operator = LocalBinaryPatterns(settings.bin_patterns_poins,
                                   settings.bin_patterns_radius)

    logger.info('Preparing patterns for images recognition...')

    training_images = get_training_images(training_dir)

    for sort, images in training_images.items():
        logger.debug('Found {1} images of "{0}" trees for training'.format(
            sort, len(images)))

    model = perform_training(operator, training_images, resize, thresholding)

    perform_recognizing(operator, model, leaf_images, resize, thresholding)


if __name__ == "__main__":
    sys.exit(main())
