import cv2
import numpy as np
import image_comp

from os.path import isfile

dest = "C:\\Users\\syeda\\Desktop\\2.G\\Teknologi\\edge_billede.png"
src = "data/Plastic/Screenshot 2023-10-31 182016.png"

base_img = cv2.imread(src)
black_img = cv2.imread(src, 0)
tmp_img = cv2.filter2D(black_img, -1, image_comp.edge_kernel)
edge_img = np.zeros_like(tmp_img)

edge_img[tmp_img > 50] = 255


cv2.imshow("Base_image", base_img)
cv2.imshow("Black_white", black_img)
cv2.imshow("Edge Image", edge_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
if not isfile(dest):
    cv2.imwrite(dest, edge_img)
