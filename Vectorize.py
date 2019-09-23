import cv2
import math
import numpy as np


class Vectorize:
    def __init__(self, vec_x, vec_y):
        self.vec_x = vec_x
        self.vec_y = vec_y

    @staticmethod
    def vec_gradient(img_gray):
        vec_x = cv2.Scharr(img_gray, cv2.CV_32F, 1, 0) / 15.36
        vec_y = cv2.Scharr(img_gray, cv2.CV_32F, 0, 1) / 15.36
        return Vectorize(vec_x, vec_y)

    def vec_magnitude(self):
        res = np.sqrt(self.vec_x ** 2 + self.vec_y ** 2)
        return (res * 255 / np.max(res)).astype(np.uint8)

    def vec_smooth(self, radius, iterations=1):
        s = 2 * radius + 1
        for _ in range(iterations):
            self.vec_x = cv2.GaussianBlur(self.vec_x, (s, s), 0)
            self.vec_y = cv2.GaussianBlur(self.vec_y, (s, s), 0)

    def vec_angle(self, i, j):
        return math.degrees(math.atan2(self.vec_y[i, j], self.vec_x[i, j]))

    def magnitude(self, i, j):
        return math.sqrt(math.hypot(self.vec_x[i, j], self.vec_y[i, j]))
