import numpy as np


def polygon_to_bbox(polygon):
    x1, y1 = np.min(polygon, axis=0)
    x2, y2 = np.max(polygon, axis=0)
    coords = [x1, y1, x2 - x1, y2 - y1]
    coords = [float(round(x,1)) for x in coords]
    return coords