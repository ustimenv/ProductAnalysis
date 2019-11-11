import argparse

from detectWrapper import DetectorWrapper


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-number',   dest='lineNumber',   type=int)
    parser.add_argument('--position',      dest='position',     type=int)
    parser.add_argument('--sampling-period', dest='samplingRate', type=int, default=1)
    parser.add_argument('--port',          dest='port',         type=int, default=-1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArgs()
    line = args.lineNumber
    position = args.position
    samplingPeriod = args.samplingRate
    socketPort = args.port

    X = DetectorWrapper(lineNumber=line,
                        positionOnLine=position,
                        samplingPeriod=samplingPeriod, port=socketPort)
    X.video()

