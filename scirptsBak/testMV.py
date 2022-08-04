import os
from tqdm import tqdm
from os.path import join
import argparse
parser = argparse.ArgumentParser(description="Alchemist Method")
parser.add_argument("--window", "-w",
                    help="The window for predict",
                    required=True,
                    type=int
                    )
parser.add_argument("--start","-s",help="The start index for test", type=int, default=0)
parser.add_argument("--type","-t", help="Which type of mv need for checking.[P,B,B_nodepend]")
args = parser.parse_args()
assert(args.window in [1, 3, 7, 15])
PredictWindow = args.window
if args.type == "P":
    videoBaseDir = "videoInfo/1I%dP" % PredictWindow
elif args.type == "B":
    videoBaseDir = "videoInfo/1P%dB" % PredictWindow
elif args.type == "B_nodepend":
    videoBaseDir = "videoInfo/1P%dB_nodepend" % PredictWindow
videoList = sorted(os.listdir(videoBaseDir))



for idx, video in enumerate(videoList):
    if idx < args.start:
        continue
    print(idx, video)
    videoDir = join(videoBaseDir, video)

    IPlist = []
    with open(join(videoDir, "IPlist.txt"), "r") as f:
        for line in f.readlines():
            line = line.strip()
            frameId, frameType = int(line.split(' ')[0]), line.split(' ')[1]
            IPlist.append((frameId, frameType))

    MVlist = {}
    with open(join(videoDir, "MotionVector.txt"), "r") as f:
        for idx, line in enumerate(f.readlines()):
            if line.strip() == '':
                continue
            try:
                dstFrameId, srcFrameId, dstX, dstY, srcX, srcY, width, height = [
                    int(x) for x in line.strip().split(',')]
            except:
                print(join(videoDir, "MotionVector.txt"), idx, line)
                exit(1)
            if srcFrameId not in MVlist:
                MVlist[srcFrameId] = []
            MVlist[srcFrameId].append(
                (dstFrameId, srcFrameId, dstX, dstY, srcX, srcY, width, height))