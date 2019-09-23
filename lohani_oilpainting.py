import bisect
import cv2
import math
import numpy as np
import Vectorize as vec
import random
import scipy.spatial

img_path = "C:\\01.jpg"
k = 30
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


def create_palette(center_base):
    center_first = center_base.copy()
    center_first[:, 1] += 50

    center_second = center_base.copy()
    center_second[:, 0] += 15
    center_second[:, 1] += 30

    center_third = center_base.copy()
    center_third[:, 0] -= 15
    center_third[:, 1] += 30

    args = (center_base, center_first, center_second, center_third)
    palette = np.concatenate(args)
    palette = np.clip(palette, 0, 255)
    return palette


def create_grid(size_h, size_w, scale):
    # Choose random coordinates in every other scale
    # i.e. if scale = 3
    # |*+*|+**|**+|*+*|+**|**+|...
    grid_h = math.ceil(size_h / scale)
    grid_w = math.ceil(size_w / scale)

    grid_random_h = np.random.randint(7, size=(grid_h, grid_w)) - 3
    grid_raw_h = np.array([[w] * grid_w for w in range(0, size_h, scale)])

    grid_random_w = np.random.randint(7, size=(grid_h, grid_w)) - 3
    grid_raw_w = np.transpose(np.array([[h] * grid_h for h in range(0, size_w, scale)]))

    args = np.transpose(np.array([(grid_random_h + grid_raw_h) % size_h, (grid_random_w + grid_raw_w) % size_w]))
    grid = args.reshape(-1, 2)
    np.random.shuffle(grid)
    grid = list(map(tuple, list(grid.copy())))
    return grid


def compute_color_probabilities(pixels, palette, k=12):
    distances = scipy.spatial.distance.cdist(pixels, palette)
    maxima = np.amax(distances, axis=1)

    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    distances = np.exp(k * len(palette) * distances)
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    return np.cumsum(distances, axis=1, dtype=np.float32)


def color_select(probabilities, palette):
    r = random.uniform(0, 1)
    i = bisect.bisect_left(probabilities, r)
    return palette[i] if i < len(palette) else palette[-1]


def main():
    img = cv2.imread(img_path)
    stroke_scale = int(math.ceil(max(img.shape) / 1000))
    gradient_smoothing_radius = int(round(max(img.shape) / 50))

    # Palette
    img_reshaped = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    z = np.float32(img_reshaped.reshape((-1, 3)))
    _, _, center_base = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    palette = create_palette(center_base)

    # Gradient
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    gradient = vec.Vectorize.vec_gradient(gray)
    gradient.vec_smooth(gradient_smoothing_radius)

    # Create a random grids(x,y) every other scale
    grid = create_grid(img.shape[0], img.shape[1], scale=3)

    # Draw
    res = cv2.medianBlur(img, 11)
    batch_size = 10000

    for h in range(0, len(grid), batch_size):
        pixels = np.array([img[x[0], x[1]] for x in grid[h:min(h + batch_size, len(grid))]])
        color_probabilities = compute_color_probabilities(pixels, palette, k=9)

        for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
            color = color_select(color_probabilities[i], palette).astype("float64")
            angle = gradient.vec_angle(y, x) + 90
            length = int(round(stroke_scale + stroke_scale * gradient.magnitude(y, x)))
            cv2.ellipse(res, (x, y), (length, stroke_scale), angle, 0, 360, color, -1, cv2.LINE_AA)

    cv2.imwrite("021.jpg", res)


if __name__ == '__main__':
    main()
