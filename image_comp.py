import cv2
import numpy as np

from statistics import mean

sub = cv2.createBackgroundSubtractorKNN()

sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
clahe = cv2.createCLAHE(0.5, (4, 4))

def mse(base_image, img1):
    h, w = base_image.shape
    tmp_img = cv2.resize(img1, (w, h))
    diff = cv2.subtract(base_image, tmp_img)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    return mse, diff

def mmse(base_img, img_list):
    count = []
    for img in img_list:
        count.append(mse(base_img, img)[0])
    return mean(count)

def object_preprocess(img):
    #tmp = cv2.medianBlur(img, 7)
    edge_filter = cv2.filter2D(img, -1, edge_kernel)

    out = edge_filter
    '''
    corners = cv2.goodFeaturesToTrack(edge_filter, 4, 0.5, 10)
    if len(corners[0]) > 3:
        dst_list = np.float32([
            [0, 0],
            [0, edge_filter[0]],
            edge_filter[0:2],
            [edge_filter[1], 0]
        ])
        for i in dst_list:
            print(i)
        matrix = cv2.getPerspectiveTransform([corners[0:5]], dst_list)
        out = cv2.warpPerspective(edge_filter, matrix, out.shape[0:2])
    else:
        out = edge_filter
    '''

    return out
def process(url):
    img = cv2.imread(url, 0)
    img = clahe.apply(img)
    return object_preprocess(img)

