import argparse

from detectWrapper import DetectorWrapper

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--line-number',   dest='lineNumber',   type=int, default=1)
    parser.add_argument('--position',      dest='position',     type=str, default='postcool')
    parser.add_argument('--sampling-rate', dest='samplingRate', type=int, default=1)
    parser.add_argument('--show-feed',     dest='showFeed',     type=int, default=0)
    parser.add_argument('--port',          dest='port',         type=int, default=-1)
    parser.add_argument('--run',           dest='run',          type=int, default=0)
    parser.add_argument('--starting-number',           dest='startNum',          type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArgs()
    line = args.lineNumber
    position = args.position
    showFeed = args.showFeed
    samplingRate = args.samplingRate
    run = bool (args.run)
    socketPort = args.port
    startNum = args.startNum

    X = DetectorWrapper(lineNumber=line,
                        position=position,
                        showFeed=showFeed,
                        samplingRate=samplingRate, run=run, port=socketPort, startNum=startNum)
    X.video()

