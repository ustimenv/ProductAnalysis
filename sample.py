import cv2
import numpy as np
import scipy.fftpack as fpack
import matplotlib.pyplot as plt
from collections import OrderedDict

from imgUtils import ImgUtils


class Desampler:
    def __init__(self):
        pass

    arr = [172, 178, 176, 170, 176, 178, 176, 172, 172, 178, 180, 164, 176, 172, 164, 172, 176, 176, 196, 170, 168,
           166, 168, 146, 86, 174, 204, 176, 174, 178, 170, 174, 170, 172, 174, 172, 170, 172, 172, 166, 160, 172, 168,
           154, 172, 172, 170, 174, 176, 178, 172, 166, 176, 178, 162, 176, 182, 174, 178, 174, 170, 170, 176, 170, 168,
           160, 170, 172, 178, 168, 174, 172, 172, 170, 172, 166, 184, 170, 154, 176, 170, 168, 164, 172, 174, 172,
           172, 170, 172, 174, 176, 172, 168, 174, 170, 170, 174, 172, 174, 190, 156, 168, 190, 172, 172, 168,
           168, 166, 172, 204, 174, 176, 170, 168, 178, 170, 180, 172, 170, 164, 164, 168, 180, 172, 174,
           160, 176, 170, 168, 172, 168, 180, 170, 178, 160, 168, 172, 164, 168, 178, 178, 180, 180, 172, 164,
           184, 172, 182, 176, 172, 232, 172, 176, 170, 178, 176, 164, 178, 174, 176, 176, 172, 172, 172, 172,
           164, 170, 168, 170, 168, 166, 186, 168, 176, 176, 176, 180, 172, 170, 170, 172, 168, 168, 172, 164, 160,
           172, 160, 176, 192, 160, 200, 186, 170, 172, 166, 174, 176, 172, 160, 172, 166, 180, 176, 168, 174, 168,
           182, 230, 172, 170, 172, 170, 174, 166, 164, 170, 162, 182, 176, 164, 172, 172, 170, 170, 178, 176, 168,
           168, 164, 174, 170, 172, 162, 180, 176, 174, 176, 168, 168, 178, 174, 170, 172, 170, 176, 178, 176, 178,
           170, 182, 92, 136, 208, 160, 122, 184, 172, 172, 170, 176, 180, 172, 160, 168, 164, 180, 170, 166, 160, 172,
           170, 172, 178, 168, 162, 184, 178, 172, 176, 168, 172, 170, 168, 166, 172, 170, 172, 164, 160, 170, 170, 172,
           202, 166, 166, 164, 170, 164, 166, 168, 160, 164, 162, 172, 164, 160, 166, 166, 172, 160, 168, 188, 188, 156,
           172, 182, 178, 174, 240, 164, 170, 178, 94, 172, 178, 84, 174, 100, 174, 180]

    def desample(self, path, batchSize=3):
        for i in range(0, 1000000, batchSize):
            imgs = []
            gmis = []
            gims = []
            for j in range(batchSize):
                imgPath = path+str(i+j)+'.png'
                img = cv2.imread(imgPath)
                if img is None:
                    continue
                img = img[60:240, 60:240, :]
                imgs.append(img)

            for j, img in enumerate(imgs):
                # imgs[0] = ImgUtils.randomImage(red=(80, 240), blue=(10, 100), green=(80, 240), imgLike=img)
                # imgs[1] = ImgUtils.randomImage(red=(0, 120), blue=(0, 120), green=(0, 120), imgLike=img)
                # imgs[2] = ImgUtils.randomImage(red=(0, 255), blue=(0, 255), green=(0, 255), imgLike=img)
                # imgs[3] = ImgUtils.randomImage(red=(120, 255), blue=(120, 255), green=(120, 255), imgLike=img)
                # imgs[4] = ImgUtils.randomImage(red=(150, 170), blue=(30, 50), green=(30, 50), imgLike=img)

                gmi = self.transform(imgs[j])
                gmis.append(gmi)
                ImgUtils.putText(img, str(np.mean(img[:, :, 2])), (10, 50), colour=(200, 0, 255))

            img = ImgUtils.hconcat(imgs, pad=50)
            gmi = ImgUtils.hconcat(gmis, pad=50)
            # Y = ImgUtils.hconcat(imgs, channel=1)
            # gmi[i, j] = (np.random.randint(10, 100), np.random.randint(70, 100), np.random.randint(80, 240))

            while True:
                ImgUtils.show('Img', img, 0, 0)
                ImgUtils.show('Gmi', gmi, 0, 260)
                # ImgUtils.show('Y', Y, 0, 720)

                key = cv2.waitKey(500)
                if key == ord('v'):
                    break
                elif key == ord('q'):
                    return

    def transform2(self, img):
        img = np.float32(img)
        fft = np.fft.fft2(img)
        f_shift = np.fft.fftshift(fft)
        f_complex = f_shift[:, :, 0] + 1j * f_shift[:, :, 1]
        f_abs = np.abs(f_complex) + 1  # lie between 1 and 1e6
        f_bounded = 20 * np.log(f_abs)
        f_img = 255 * f_bounded / np.max(f_bounded)
        f_img = f_img.astype(np.uint8)
        return f_img

    def transform(self, img):
        h, w, _ = img.shape
        img = np.float32(img)
        fft = np.fft.fft(img, axis=0)
        fft = np.fft.fftshift(fft)
        fft = 20 * np.log(np.abs(fft))
        fft = np.uint8(abs(fft))
        return fft

    def dimAnalysis(self):
        R = OrderedDict()
        for r in sorted(self.arr):
            if r in R.keys():
                R[r]+=1
            else:
                R[r] = 1

        print(R, sep='\n')

        plt.bar(range(len(R)), list(R.values()), align='center')
        plt.xticks(range(len(R)), list(R.keys()))
        plt.show()

    def getFoo(self, img=None, flag=1):
        if flag == 1:
            return np.zeros_like(img)*255

        elif flag == 2:
            return np.ones_like(img)*255

        elif flag == 3:
            h, w, c = img.shape
            a = np.zeros((int(h/2), w, c), dtype=img.dtype)
            b = np.ones((int(h/2), w, c), dtype=img.dtype)
            img = np.vstack((b, a))*255

            return img
        elif flag == 4:
            h, w, _ = img.shape
            gmi = np.zeros_like(img)
            for i in range(h):
                for j in range(w):
                    if i < j:
                        gmi[i, j] = 255
                    else:
                        gmi[i, j] = 0
            return gmi
        elif flag == 5:
            h, w, _ = img.shape
            gmi = np.zeros_like(img)
            for i in range(h):
                for j in range(w):
                    if i > j:
                        gmi[i, j] = 255
                    else:
                        gmi[i, j] = 0
            return gmi


if __name__=="__main__":
    D = Desampler()
    D.dimAnalysis()
    # D.desample('/beta/Work/2/Train/1/')