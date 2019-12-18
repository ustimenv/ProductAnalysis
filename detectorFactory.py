import importlib

INTMAX = 2** 60


class DetectorFactory:
    """
        :::dim1: Rectangles->width;     Circles->radius
        :::dim2: Rectangles->height;    Circles->not applicable
    """
    args = \
        {
            (1, 0): {'dim1Lower': 120, 'dim1Upper': 300,
                         'dim2Lower': 50,  'dim2Upper': 200,
                         'upperKillzone': INTMAX, 'lowerKillzone': -INTMAX,
                         'rightKillzone': INTMAX, 'leftKillzone' : 50,
                         'timeToDie': 1, 'timeToLive': 0,
                         'name'         : 'raw1',
                         'roiTrackingMode': True,
                         'initialRoi' : (250, 300, 760, 670),
                         'dimensionTracking' : False,
                         'colourTracking'    : False
                         },

            (1, 1): {'dim1Lower': 40, 'dim1Upper': 150,
                              'dim2Lower': None,  'dim2Upper': None,
                              'upperKillzone': 300, 'lowerKillzone': 220,
                              'rightKillzone': 270, 'leftKillzone': 30,
                              'timeToDie': 1, 'timeToLive': 0,
                              'name' : 'postbake1',
                              'roiTrackingMode': False,
                              'initialRoi' : (530, 150, 830, 1000),
                                 'dimensionTracking': True,
                                 'colourTracking': True
                              },

            (3, 0): {'dim1Lower': 140, 'dim1Upper': 350,
                             'dim2Lower': 140,  'dim2Upper': 550,
                             'upperKillzone': INTMAX, 'lowerKillzone': -INTMAX,
                             'rightKillzone': INTMAX, 'leftKillzone' : -INTMAX,
                             'timeToDie': 5, 'timeToLive': 0,
                             'name'         : 'brick0',
                             'roiTrackingMode': True,
                             'initialRoi' : (0, 0, 0, 0),
                             'dimensionTracking': False,
                             'colourTracking': False
                         }
        }

    def __init(self):
        pass

    @staticmethod
    def getDetector(line, position):
        line = 'Line'+str(line)
        detector = line+'/detect'+str(position)+'.py'

        from importlib.machinery import SourceFileLoader
        foo = SourceFileLoader(line, detector).load_module()
        return foo.Detector()

    # @staticmethod
    # def getDetector(lineNumber, positionOnLine):
    #     return Detector(**DetectorFactory.args[(lineNumber, positionOnLine)])

