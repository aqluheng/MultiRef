import torch
import os
from os.path import join
import xml.dom.minidom
import copy
from tqdm import tqdm

from mega_core.structures.bounding_box import BoxList

PredictWindow = 15
baseDir = "1I%dP" % PredictWindow
videoList = sorted(os.listdir(baseDir))
visualDir = "visual/"
outputDir = "Alchemist1I%dP/" % PredictWindow

labelDict = {"n02691156": 1, "n02419796": 2, "n02131653": 3, "n02834778": 4, "n01503061": 5, "n02924116": 6, "n02958343": 7,
             "n02402425": 8, "n02084071": 9, "n02121808": 10, "n02503517": 11, "n02118333": 12, "n02510455": 13, "n02342885": 14, "n02374451": 15,
             "n02129165": 16, "n01674464": 17, "n02484322": 18, "n03790512": 19, "n02324045": 20, "n02509815": 21, "n02411705": 22,
             "n01726692": 23, "n02355227": 24, "n02129604": 25, "n04468005": 26, "n01662784": 27, "n04530566": 28, "n02062744": 29, "n02391049": 30}


def visualVideo(video, videoBBox):
    import cv2
    os.system("mkdir -p %s" % visualDir)
    imgDir = "ILSVRC2015/Data/VID/val/%s" % video
    for idx, BBoxList in enumerate(videoBBox):
        img = cv2.imread(join(imgDir, "%06d.JPEG" % idx))
        for bbox in BBoxList.bbox:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(
                bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.imwrite(join(visualDir, "%06d.JPEG" % idx), img)

allVideoBBox = []
os.system("mkdir -p %s" % outputDir)
for video in tqdm(videoList):
    videoDir = join(baseDir, video)

    IPlist = []
    with open(join(videoDir, "IPlist.txt"), "r") as f:
        for line in f.readlines():
            line = line.strip()
            frameId, frameType = int(line.split(' ')[0]), line.split(' ')[1]
            IPlist.append((frameId, frameType))

    MVlist = {}
    with open(join(videoDir, "MotionVector.txt"), "r") as f:
        for line in f.readlines():
            dstFrameId, srcFrameId, dstX, dstY, srcX, srcY, width, height = [
                int(x) for x in line.split(',')]
            if srcFrameId not in MVlist:
                MVlist[srcFrameId] = []
            MVlist[srcFrameId].append(
                (dstFrameId, srcFrameId, dstX, dstY, srcX, srcY, width, height))

    BBoxList = [None] * len(IPlist)

    def readFromXML(frameId):

        annoDir = "ILSVRC2015/Annotations/VID/val/%s" % video
        dom = xml.dom.minidom.parse(
            join(annoDir, "%06d.xml" % frameId)).documentElement
        width = dom.getElementsByTagName("width")[0].childNodes[0].data
        height = dom.getElementsByTagName("height")[0].childNodes[0].data
        objectLists = dom.getElementsByTagName("object")
        allBBox = []
        scores = []
        labels = []
        for BBoxObject in objectLists:
            xmin = int(BBoxObject.getElementsByTagName(
                "xmin")[0].childNodes[0].data)
            ymin = int(BBoxObject.getElementsByTagName(
                "ymin")[0].childNodes[0].data)
            xmax = int(BBoxObject.getElementsByTagName(
                "xmax")[0].childNodes[0].data)
            ymax = int(BBoxObject.getElementsByTagName(
                "ymax")[0].childNodes[0].data)
            name = BBoxObject.getElementsByTagName(
                "name")[0].childNodes[0].data
            allBBox.append([xmin, ymin, xmax, ymax])
            scores.append(1)
            labels.append(labelDict[name])
        if allBBox == []:
            allBBox.append([0, 0, 0, 0])
            scores.append(0)
            labels.append(1)

        bboxList = BoxList(allBBox, (width, height))
        bboxList.add_field("scores", torch.Tensor(scores))
        bboxList.add_field("labels", torch.Tensor(labels).int())
        return bboxList

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
            BBoxList[frameId] = readFromXML(frameId)
            # print(frameId, BBoxList[frameId].bbox)
            mapBBox(frameId)
            # print(frameId,BBoxList[frameId].bbox)

    # visualVideo(video, BBoxList)
    torch.save(BBoxList, join(outputDir, "%s.pth" % video))
    allVideoBBox += BBoxList
torch.save(allVideoBBox, join(outputDir, "Alchemist1I%dP.pth"%PredictWindow))
