# -*- coding: utf-8 -*-
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

import cv2


class SegmentedImage(object):

    def __init__(self, gray_image):
        self.gray_image = gray_image
        _, self.segmented_image = cv2.threshold(
            self.gray_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    def __getattr__(self, item):
        self.segmented_image.__getattr__(item)

    def __getitem__(self, item):
        self.segmented_image.__getitem__(item)


class ResisedImage(object):

    def __init__(self, image, size):
        self.original_image = image
        self.resized_image = cv2.resize(self.original_image, size)

    def __getattr__(self, item):
        self.resized_image.__getattr__(item)

    def __getitem__(self, item):
        self.resized_image.__getitem__(item)