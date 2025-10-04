import cv2
import pygame
import numpy as np
import image_comp


from os import listdir

print(cv2.__version__)

pygame.init()

file=0

screen = pygame.display.set_mode((900, 600))
clock = pygame.time.Clock()

FPS = 144
recordLowest = None

# Video Feed
vid = cv2.VideoCapture(file)
if not vid.isOpened():
    print('Cannot read live feed')
    exit(-1)

'''
bgSub = cv2.createBackgroundSubtractorKNN()


bgSub.setDetectShadows(False)
bgSub.setHistory(FPS)

def image_thresholding(image):
    mask = bgSub.apply(image)
    out = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)[1]
    return out
'''

class WasteType():
    def __init__(self, w_type, data):
        self.type = w_type
        self.list = None
        self.objs = None
        self.mean = None
        self.wd = data

        self.boxColor = (0, 0, 0)
        self.textColor = (0, 0, 0)



    def add(self, grp):
        grp.append(self)

    def create(self, grp):
        self.list = listdir(self.wd)
        self.objs = [image_comp.process(f"{self.wd}/{w}") for w in self.list]
        self.mean = image_comp.mmse(self.objs[0], self.objs)
        self.add(grp)

    def setColor(self, box, text):
        self.boxColor = box
        self.textColor = text
    def setBox(self):
        return (self.type, self.boxColor, self.textColor)


w_types = []

w_plastic = WasteType("Plastic", "data/Plastic")
w_non_rc = WasteType("Non-Recyclable Waste", None)
#w_papPapir = WasteType("Paper or Carton", "data/Pap-og-papir")
w_metal = WasteType("Metal", "data/Metal")

w_plastic.setColor((0, 0, 255), (255, 255, 255))
#w_papPapir.setColor((255, 0, 0), (0, 0, 255))
w_metal.setColor((0, 255, 255), (255, 255, 255))
w_non_rc.setColor((0, 255, 0), (255, 0, 0))

w_plastic.create(w_types)
w_metal.create(w_types)
#w_papPapir.create(w_types)


boxColor = (0, 255, 0)
textColor = (255, 0, 0)


while True:
    # Video Parsing
    ret, frame = vid.read()
    if not ret:
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)

    output = frame.copy()

    frame = cv2.medianBlur(frame, 11)
    cl1 = image_comp.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    #buff = cv2.threshold(cv2.filter2D(cl1, -1, image_comp.edge_kernel), 60, 255, cv2.THRESH_BINARY)[1]
    edge_filter = cv2.filter2D(cl1, -1, image_comp.edge_kernel)
    buff = np.zeros_like(edge_filter)
    buff[edge_filter > 18] = 255

    contours, hierarchy = cv2.findContours(buff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        for (c, hr) in zip(contours, hierarchy[0]):
            if hr[3] < 0:
                x, y, w, h = cv2.boundingRect(c)
                if w+h > 50 and w+h < (screen.get_width()+screen.get_height())//2:
                    roi = buff[y:y+h, x:x+w]
                    corners = cv2.goodFeaturesToTrack(roi, 4, 0.5, 20)

                    if type(corners) == 'numpy.ndarray':
                        if len(corners[0]) > 3:
                            dst_list = np.float32([
                                [0, 0],
                                [0, roi.shape[0]],
                                roi.shape[0:2],
                                [roi.shape[1], 0]]
                            )

                            M = cv2.getPerspectiveTransform([corners[0:5]], dst_list)
                            roi = cv2.warpPerspective(roi, M, roi.shape[0:2])
                            #roi = cv2.fastNlMeansDenoising(roi, 3)

                    tmp = []

                    for waste in w_types:
                        if type(waste) != 'NoneType':
                            tmp.append(image_comp.mmse(roi, waste.objs))
                            print(f"{waste.type}: {tmp[-1]}")

                    i = tmp.index(min(tmp))


                    '''
                    i = 1
                    if image_comp.mmse(roi, w_plastic.objs) < 1.75:
                        i = 0
                    '''

                    nameTag, boxColor, textColor = w_types[i].setBox() if min(tmp) < 2.15 else w_non_rc.setBox()

                    cv2.rectangle(output, (x, y - 20), (len(nameTag)*14 + x, y), boxColor, -1)
                    cv2.putText(output, nameTag, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, color=textColor, thickness=2)
                    cv2.rectangle(output, (x, y), (w + x, h + y), boxColor, 4)


    buff = cv2.cvtColor(buff, cv2.COLOR_GRAY2BGR)

    # Subtracting background from picture.
    fgMask = pygame.image.frombuffer(buff.tobytes(), buff.shape[1::-1], "BGR")
    fgMask = pygame.transform.scale(fgMask, (screen.get_width()//2, screen.get_height()))

    bg = pygame.image.frombuffer(output.tobytes(), output.shape[1::-1], "BGR")
    bg = pygame.transform.scale(bg, fgMask.get_size())

    screen.blit(fgMask, (0, 0))
    screen.blit(bg, (screen.get_width()//2, 0))

    pygame.display.update()
    clock.tick(FPS)
