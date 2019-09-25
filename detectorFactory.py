from detect import Detector

INTMAX = 2** 60


class DetectorFactory:
    """
        :::dim1: Rectangles->width;     Circles->radius
        :::dim2: Rectangles->height;    Circles->not applicable
    """
    args = \
        {
            (1, 'raw'): {'dim1Lower': 140, 'dim1Upper': 350,
                         'dim2Lower': 140,  'dim2Upper': 550,
                         'upperKillzone': INTMAX, 'lowerKillzone': -INTMAX,
                         'rightKillzone': INTMAX, 'leftKillzone' : -INTMAX,
                         'timeToDie': 1, 'timeToLive': 0,
                         'name'         : 'raw1',
                         'roiTrackingMode': True,
                         'initialRoi' : (250, 300, 760, 670)
                         },

            (1, 'postbake'): {'dim1Lower': 40, 'dim1Upper': 150,
                              'dim2Lower': None,  'dim2Upper': None,
                              'upperKillzone': 550, 'lowerKillzone': 220,
                              'rightKillzone': INTMAX, 'leftKillzone': -INTMAX,
                              'timeToDie': 1, 'timeToLive': 0,
                              'name' : 'postbake1',
                              'roiTrackingMode': False,
                              'initialRoi' : (530, 150, 830, 1000)
                              },

            (1, 'postbakeDebug'): {'dim1Lower': None, 'dim1Upper': None,
                                   'dim2Lower': None,  'dim2Upper': None,
                                   'upperKillzone': 0, 'lowerKillzone': 0,
                                   'rightKillzone': 0, 'leftKillzone': 0,
                                   'timeToDie': 1, 'timeToLive': 0,
                                   'roiTrackingMode': False
                                   }
        }

    def __init(self):
        pass

    @staticmethod
    def getDetector(lineNumber, positionOnLine):
        return Detector(**DetectorFactory.args[(lineNumber, positionOnLine)])

