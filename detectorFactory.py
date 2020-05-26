

class DetectorFactory:

    def __init(self):
        pass

    @staticmethod
    def getDetector(line, position):
        line = 'Line'+str(line)
        detector = line+'/detect'+str(position)+'.py'

        from importlib.machinery import SourceFileLoader
        foo = SourceFileLoader(line, detector).load_module()
        return foo.Detector()

