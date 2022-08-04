import xml.dom.minidom
from os.path import join
import os
import torch

from mega_core.structures.bounding_box import BoxList
# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


labelDict = {"n02691156": 1, "n02419796": 2, "n02131653": 3, "n02834778": 4, "n01503061": 5, "n02924116": 6, "n02958343": 7,
             "n02402425": 8, "n02084071": 9, "n02121808": 10, "n02503517": 11, "n02118333": 12, "n02510455": 13, "n02342885": 14, "n02374451": 15,
             "n02129165": 16, "n01674464": 17, "n02484322": 18, "n03790512": 19, "n02324045": 20, "n02509815": 21, "n02411705": 22,
             "n01726692": 23, "n02355227": 24, "n02129604": 25, "n04468005": 26, "n01662784": 27, "n04530566": 28, "n02062744": 29, "n02391049": 30}

gtPrediction = []
for video in sorted(os.listdir("ILSVRC2015/Annotations/VID/val/")):
    annoDir = "ILSVRC2015/Annotations/VID/val/%s" % video
    infoDir = "videoInfo/%s" % video
    for fileName in sorted(os.listdir(annoDir)):
        # 左上角右下角与图片大小
        dom = xml.dom.minidom.parse(join(annoDir, fileName)).documentElement
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
        gtPrediction.append(bboxList)
torch.save(gtPrediction, "gtPrediction.pth")
