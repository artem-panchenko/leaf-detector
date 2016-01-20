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


import argparse
from logging import DEBUG
import os
import sys

import errors
from logger import logger


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Detector types of plans by photos of their trees."
    )
    parser.add_argument('leaf_image', type=str, nargs='+',
                        help="Path to leaf image to recognize")
    parser.add_argument("-t", "--training", required=True,
                        help="Path to the directory with training images.")
    parser.add_argument('-T', '--thresholding', default=False,
                        action='store_true', help='Enable images thresholding')
    parser.add_argument('-r', '--resize', default=False,  action='store_true',
                        help='Enable images resizing')
    parser.add_argument('-v', '--verbose', default=False,  action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(DEBUG)

    if not all(os.path.isfile(leaf) for leaf in args.leaf_image):
        raise errors.LeadDetectionError('Not found images: {0}!'.format(
            args.leaf_image))

    if not os.path.isdir(args.training):
        raise errors.LeadDetectionError('Directory {0} doesn\'t exist!'.format(
            args.training))

    return args.leaf_image, args.training, args.thresholding, args.resize
