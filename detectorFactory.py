from detect import Detector

INTMAX = 2** 60


class DetectorFactory:
    """
        :::dim1: Rectangles->width;     Circles->radius
        :::dim2: Rectangles->height;    Circles->not applicable
    """
    args = \
        {
            (1, 'raw'): {'dim1Lower': 120, 'dim1Upper': 300,
                         'dim2Lower': 50,  'dim2Upper': 200,
                         'upperKillzone': INTMAX, 'lowerKillzone': -INTMAX,
                         'rightKillzone': INTMAX, 'leftKillzone' : 50,
                         'timeToDie': 1, 'timeToLive': 0,
                         'name'         : 'raw1',
                         'roiTrackingMode': True,
                         'initialRoi' : (250, 300, 760, 670)
                         },

            (1, 'postbake'): {'dim1Lower': 40, 'dim1Upper': 150,
                              'dim2Lower': None,  'dim2Upper': None,
                              'upperKillzone': 300, 'lowerKillzone': 220,
                              'rightKillzone': 270, 'leftKillzone': 30,
                              'timeToDie': 1, 'timeToLive': 0,
                              'name' : 'postbake1',
                              'roiTrackingMode': False,
                              'initialRoi' : (530, 150, 830, 1000)
                              },

            (3, 'brick'): {'dim1Lower': 140, 'dim1Upper': 350,
                             'dim2Lower': 140,  'dim2Upper': 550,
                             'upperKillzone': INTMAX, 'lowerKillzone': -INTMAX,
                             'rightKillzone': INTMAX, 'leftKillzone' : -INTMAX,
                             'timeToDie': 5, 'timeToLive': 0,
                             'name'         : 'brick3',
                             'roiTrackingMode': True,
                             'initialRoi' : (0, 0, 0, 0)
                         }
        }

    def __init(self):
        pass

    @staticmethod
    def getDetector(lineNumber, positionOnLine):
        return Detector(**DetectorFactory.args[(lineNumber, positionOnLine)])

