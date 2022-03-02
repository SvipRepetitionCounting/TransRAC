""" transform label to groundtruth(density map)"""
from scipy import integrate
import math
import numpy as np


def PDF(x, u, sig):
    # f(x)
    return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)

# integral f(x)
def get_integrate(x_1, x_2, avg, sig):
    y, err = integrate.quad(PDF, x_1, x_2, args=(avg, sig))
    return y


def normalize_label(y_frame, y_length):
    # y_length: total frames
    # return: normalize_label  size:nparray(y_length,)
    y_label = [0 for i in range(y_length)]  # 坐标轴长度，即帧数
    for i in range(0, len(y_frame), 2):
        x_a = y_frame[i]
        x_b = y_frame[i + 1]
        avg = (x_b + x_a) / 2
        sig = (x_b - x_a) / 6
        num = x_b - x_a + 1  # 帧数量 update 1104
        if num != 1:
            for j in range(num):
                x_1 = x_a - 0.5 + j
                x_2 = x_a + 0.5 + j
                y_ing = get_integrate(x_1, x_2, avg, sig)
                y_label[x_a + j] = y_ing
        else:
            y_label[x_a] = 1
    return y_label
