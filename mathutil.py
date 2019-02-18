import numpy as np


def matrix_cut(matrix, pos, direciton):
    x, y = pos[0], pos[1]
    dx, dy = direciton[0], direciton[1]
    ret = []
    t = 0
    if dx == 0 and dy == 0:
        return np.zeros(1, dtype='int8')

    try:
        while True:
            if x + dx * t < 0 or y + dy * t < 0:
                break
            ret.append(matrix[x + dx * t, y + dy * t])
            t += 1
            if matrix[x + dx * t, y + dy * t] == 0:
                break

    except IndexError:
        pass
    return np.array(ret, dtype='int8')
