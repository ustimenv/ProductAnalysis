import cv2
import numpy as np
from skimage.measure import compare_ssim


class ImgUtils:

    @staticmethod
    def show(windowName, contents, xCoordinate, yCoordinate):
        image = None

        if(type(contents) == str) :
            image = cv2.imread("contents")
        else:
            image = contents
        cv2.namedWindow(windowName)
        cv2.moveWindow(windowName, xCoordinate, yCoordinate)
        cv2.imshow(windowName, image)

    @staticmethod
    def compareImages(A, B):
        (score, diff) = compare_ssim(A, B, full=True)
        diff = (diff * 255).astype("uint8")
        return diff

    @staticmethod
    def boxArea(x1,y1,x2,y2):
        return (x2-x1) * (y2-y1)

    @staticmethod
    def getCentroid(box):
        cX = int((box[0] + box[2]) / 2.0)
        cY = int((box[1] + box[3]) / 2.0)
        return cX, cY

    @staticmethod
    def drawRect(rect, frame, colour=(255,0,0), offset=(0,0,0,0)):
        x1 = int(rect[0])+offset[0]
        y1 = int(rect[1])+offset[1]
        x2 = int(rect[2])+offset[2]
        y2 = int(rect[3])+offset[3]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 4)

    @staticmethod
    def drawCircle(centre, img, radius=6, colour=(0,0,255)):
        cv2.circle(img=img, center=centre, radius=radius, thickness=4, color=colour)

    @staticmethod
    def putText(img, text, coords, xOffset=0, yOffset=0, colour=(0, 255, 255), thickness=2, fontSize=5):

        cv2.putText(img=img, text=text, org=(int(coords[0]+xOffset), int(coords[1]+yOffset)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, thickness=thickness, lineType=cv2.LINE_4, fontScale=fontSize, color=colour)

    @staticmethod
    def returnUnchanged(x):
        return x

    @staticmethod
    def circleToRectabgle(center, radius):
        cX = int(center[0]); cY = int(center[1]); r = int(radius)
        xmin = cX-r; ymin = cY-r; xmax = cX + r; ymax = cY +r
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def sampleRoi(img, roi, sampleDims=None):
        """

        :param img:
        :param roi:
        :param sampleDims:   (ySize, xSize)
        :return:
        """
        if sampleDims is None:
            return img[roi[1]:roi[3], roi[0]:roi[2]]
        else:
            dy = int(sampleDims[0]/2)
            dx = int(sampleDims[1]/2)
            cX, cY = ImgUtils.getCentroid(roi)
            return img[cY-dy : cY+dy, cX-dx:cX+dx]

    @staticmethod
    def sampleAround(img, cX, cY, width, height):
        dy = int(height/2)
        dx = int(width/2)
        return img[cY-dy : cY+dy, cX-dx:cX+dx]

    @staticmethod
    def showLines(img, lines):
        if lines is not None:
            for line in lines:
                rho = line[0][0]
                theta = line[0][1]
                a = np.math.cos(theta)
                b = np.math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
        return img

    @staticmethod
    def hconcat(imgs, channel=None, pad=0):
        if pad != 0:
            h, w, c = imgs[0].shape
            buffer = np.zeros(shape=(h, pad, c), dtype=imgs[0].dtype)
            for i in range(len(imgs)-1):
                imgs[i] = cv2.hconcat((imgs[i], buffer))

        if channel is not None:
            for i, img in enumerate(imgs):
                imgs[i] = img[:, :, channel, np.newaxis]

        return cv2.hconcat(imgs)



    @staticmethod
    def randomImage(red, blue, green, imgLike=None, dims=None, dtype=None):
        if imgLike is not None:
            h, w, c = imgLike.shape
            gmi = np.zeros_like(imgLike)
        else:
            h, w, c = dims
            gmi = np.zeros((h, w, c), dtype=dtype)

        for i in range(h):
            for j in range(w):
                gmi[i, j] = (np.random.randint(blue[0], blue[1]),
                             np.random.randint(green[0], green[1]),
                             np.random.randint(red[0], red[1]))
        return gmi


if __name__ == "__main__":
    # SystemUtils.initDir('DataX', scaleFactor=5)
    # SystemUtils.copySome()
    # for imgName in glob.glob('DataReduced/*.png', recursive=False):
    #     plt.imshow(cv2.imread(imgName))
    #     print(imgName)
    #     plt.show()
    # SystemUtils.getExecutionRate()
    pass