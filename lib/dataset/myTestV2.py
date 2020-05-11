pairs = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16],
         [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]]

# print(pairs[0][0])

import cv2
import numpy as np

# map = np.zeros((64, 48), dtype=np.uint8)
# print(map.shape)
# cv2.line(map, (20, 20), (21, 21), (255, 0, 0), 1)
# cv2.circle(map, (10, 10), 1, (255, 0, 0), 1)
# cv2.applyColorMap(map, cv2.COLORMAP_JET)
# cv2.imshow('map', map)
# cv2.waitKey(0)

# map = np.zeros((640, 480), dtype=np.uint8)
# print(map.shape)
# cv2.line(map, (20, 20), (30, 30), (255, 0, 0), 3)
# cv2.circle(map, (100, 100), 10, (150, 0, 0), 1)
# cv2.circle(map, (100, 100), 9, (175, 0, 0), 1)
# cv2.circle(map, (100, 100), 8, (200, 0, 0), 1)
# cv2.circle(map, (100, 100), 7, (225, 0, 0), 1)
# cv2.circle(map, (100, 100), 6, (255, 0, 0), 1)
# map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
# cv2.imshow('map', map)
# cv2.waitKey(0)

# sigma = 2
# tmp_size = sigma * 3
# image_size = np.array((192, 256))
# heatmap_size = np.array((48, 64))
# target = np.zeros((heatmap_size[1],
#                    heatmap_size[0]),
#                   dtype=np.float32)
# feat_stride = image_size / heatmap_size
# mu_x = int(96 / feat_stride[0] + 0.5)  # +0.5就可以四舍五入了
# mu_y = int(128 / feat_stride[1] + 0.5)
# # Check that any part of the gaussian is in-bounds
# ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
# br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
# # Generate gaussian
# size = 2 * tmp_size + 1  # 直径
# x = np.arange(0, size, 1, np.float32)
# y = x[:, np.newaxis]
# x0 = y0 = size // 2  # 中心
# # The gaussian is not normalized, we want the center value to equal 1
# g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
#
# # Usable gaussian range
# g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
# g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
# # Image range
# img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
# img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
#
# target[img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
#     g[g_y[0]:g_y[1], g_x[0]:g_x[1]]  # 就是g的全部了
#
# cv2.imshow('map0', target)
# target = target * 255
# target = np.clip(target, 0, 255)
# # target[0] = np.array(target[0], np.uint8)
# target = target.astype(np.uint8)
# cv2.imshow('map1', target)
# map2 = cv2.applyColorMap(target, cv2.COLORMAP_JET)
# cv2.imshow('map2', map2)
# cv2.waitKey(0)


a = np.zeros((1, 2, 3))
print(a.shape[1:])