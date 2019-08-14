import cv2
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
        return (cX, cY)

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
    def putText(img, text, coords, xOffset=0, yOffset=0, colour=(0, 255, 255)):
        cv2.putText(img=img, text=text, org=(int(coords[0]+xOffset), int(coords[1]+yOffset)),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, thickness=1, lineType=cv2.LINE_4, fontScale=5, color=colour)


if __name__ == "__main__":
    # SystemUtils.initDir('DataX', scaleFactor=5)
    # SystemUtils.copySome()
    # for imgName in glob.glob('DataReduced/*.png', recursive=False):
    #     plt.imshow(cv2.imread(imgName))
    #     print(imgName)
    #     plt.show()
    # SystemUtils.getExecutionRate()
    pass