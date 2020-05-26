import cv2

from detect import Detector
from imgUtils import ImgUtils

if __name__ == "__main__":
    param = {(1, 'postbake') : {'expectedWidth' : 120, 'expectedHeight'     : 110,     # detection params
                           'transformationTarget'             : 'cool',  # select correct image transformations
                           'upperKillzone'  : 550, 'lowerKillzone' : 220,     # select correct tracking parameters
                           'rightKillzone'  : 3000, 'leftKillzone' : -3000,     # select correct tracking parameters
                           'timeToDie' : 1, 'timeToLive'     : 0,
                           'partioningRequired': True
                          }}

    D = Detector(**param.get((1, 'postbake')))

    for i in range(49, 100):
        img = cv2.imread("PostBakeSamples/postbake"+str(i)+".png")
        x = D.getImgWithBoxes(img)
        while True:
            ImgUtils.show("x", x, 0, 0)
            key = cv2.waitKey(30)
            if key == ord('q'):
                break
