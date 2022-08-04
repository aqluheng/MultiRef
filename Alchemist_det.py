import torch
import os
from os.path import join
import xml.dom.minidom
import copy
from tqdm import tqdm

from mega_core.structures.bounding_box import BoxList


import argparse
parser = argparse.ArgumentParser(description="Alchemist Method")
parser.add_argument("--model", "-m",
                    help="The model which generate the detection results.",
                    required=True
                    )

parser.add_argument("--window", "-w",
                    help="The window for predict",
                    required=True,
                    type=int
                    )
parser.add_argument("--visual",
                    help="Visualize BBoxes.", action="store_true")
parser.add_argument("--start","-s",help="The start index for test", type=int, default=0)

args = parser.parse_args()

assert(args.model in ["DFF", "FGFA", "MEGA", "GT"])
assert(args.window in [1, 3, 7, 15])

PredictWindow = args.window
videoBaseDir = "videoInfo/1I%dP" % PredictWindow
videoList = sorted(os.listdir(videoBaseDir))
visualDir = "AlchemistOut/base%s_visu/1I%dP" % (args.model, PredictWindow)
outputDir = "AlchemistOut/base%s/1I%dP/" % (args.model, PredictWindow)
# exit(1)

allVideoBBox = []


labelDict = {"n02691156": 1, "n02419796": 2, "n02131653": 3, "n02834778": 4, "n01503061": 5, "n02924116": 6, "n02958343": 7,
             "n02402425": 8, "n02084071": 9, "n02121808": 10, "n02503517": 11, "n02118333": 12, "n02510455": 13, "n02342885": 14, "n02374451": 15,
             "n02129165": 16, "n01674464": 17, "n02484322": 18, "n03790512": 19, "n02324045": 20, "n02509815": 21, "n02411705": 22,
             "n01726692": 23, "n02355227": 24, "n02129604": 25, "n04468005": 26, "n01662784": 27, "n04530566": 28, "n02062744": 29, "n02391049": 30}


def visualVideo(video, videoBBox):
    import cv2
    os.system("mkdir -p %s/%s" % (visualDir, video))
    imgDir = "ILSVRC2015/Data/VID/val/%s" % video
    for idx, BBoxList in enumerate(videoBBox):
        img = cv2.imread(join(imgDir, "%06d.JPEG" % idx))
        for bbox in BBoxList.bbox:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(
                bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.imwrite(join("%s/%s/" %
                    (visualDir, video), "%06d.JPEG" % idx), img)


if args.model == "DFF":
    refBBoxList = torch.load("annoPth/DFF_50_predictions.pth")
elif args.model == "FGFA":
    refBBoxList = torch.load("annoPth/FGFA_50_predictions.pth")
elif args.model == "MEGA":
    refBBoxList = torch.load("annoPth/MEGA_50_predictions.pth")
elif args.model == "GT":
    refBBoxList = torch.load("annoPth/GT_predictions.pth")

os.system("mkdir -p %s" % outputDir)

currentBaseIdx = 0

for idx ,video in tqdm(enumerate(videoList)):
    videoDir = join(videoBaseDir, video)

    IPlist = []
    with open(join(videoDir, "IPlist.txt"), "r") as f:
        for line in f.readlines():
            line = line.strip()
            frameId, frameType = int(line.split(' ')[0]), line.split(' ')[1]
            IPlist.append((frameId, frameType))
    if idx < args.start:
        currentBaseIdx += len(IPlist)
        continue
    MVlist = {}
    with open(join(videoDir, "MotionVector.txt"), "r") as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            try:
                dstFrameId, srcFrameId, dstX, dstY, srcX, srcY, width, height = [
                int(x) for x in line.strip().split(',')]
                if srcFrameId not in MVlist:
                    MVlist[srcFrameId] = []
                MVlist[srcFrameId].append(
                    (dstFrameId, srcFrameId, dstX, dstY, srcX, srcY, width, height))
            except:
                pass
            
    BBoxList = [None] * len(IPlist)

    def readBBox(frameId):
        return refBBoxList[currentBaseIdx+frameId]

    def mapBBox(frameId):
        def mergeBBox(BBox1, BBox2):
            resBBox = [None] * 4
            resBBox[0] = min(BBox1[0], BBox2[0])
            resBBox[1] = min(BBox1[1], BBox2[1])
            resBBox[2] = max(BBox1[2], BBox2[2])
            resBBox[3] = max(BBox1[3], BBox2[3])
            return torch.Tensor(resBBox)

        def movBBox(BBox1, dstX, dstY, srcX, srcY):
            resBBox = [None] * 4
            resBBox[0] = BBox1[0] + dstX - srcX
            resBBox[2] = BBox1[2] + dstX - srcX
            resBBox[1] = BBox1[1] + dstY - srcY
            resBBox[3] = BBox1[3] + dstY - srcY
            return resBBox

        for predictFrame in range(PredictWindow):
            if frameId + predictFrame + 1 >= len(BBoxList):
                break
            BBoxList[frameId+predictFrame+1] = copy.deepcopy(BBoxList[frameId])
        if frameId not in MVlist:
            return
        for dstFrameId, srcFrameId, dstX, dstY, srcX, srcY, width, height in MVlist[frameId]:
            for bboxId, bbox in enumerate(BBoxList[frameId].bbox):
                if bbox[0] - width <= srcX <= bbox[2] and bbox[1] - height <= srcY <= bbox[3]:
                    BBoxList[dstFrameId].bbox[bboxId] = mergeBBox(
                        BBoxList[dstFrameId].bbox[bboxId], movBBox(bbox, dstX, dstY, srcX, srcY))

    for frameId, frameType in IPlist:
        if frameType == "I":
            BBoxList[frameId] = readBBox(frameId)
            mapBBox(frameId)

    if args.visual:
        visualVideo(video, BBoxList)
    torch.save(BBoxList, join(outputDir, "%s.pth" % video))
    allVideoBBox += BBoxList
    currentBaseIdx += len(IPlist)

torch.save(allVideoBBox, join(outputDir, "Alchemist1I%dP.pth" % PredictWindow))
