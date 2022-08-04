import cv2
import numpy as np
import os
from os.path import join



videoDir = "ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000005"
# assert(os.path.exists(videoDir))
# os.system("./FFmpeg/ffmpeg -i %s/%%06d.JPEG -c:v libx265 tmp.mkv"%videoDir)
# os.system("./FFmpeg/ffmpeg -i tmp.mkv tmp.mp4")
# os.system("rm tmp.m*")

IPlist = []
with open("IPlist.txt", "r") as f:
    for line in f.readlines():
        frameId, frameType = [int(x) for x in line.split(' ')]
        IPlist.append((frameId, frameType))

MVlist = {}
with open("MotionVector.txt", "r") as f:
    for line in f.readlines():
        dstFrameId, srcFrameId, dstX, dstY, srcX, srcY, width, height = [
            int(x) for x in line.split(',')]
        if dstFrameId not in MVlist:
            MVlist[dstFrameId] = []
        MVlist[dstFrameId].append(
            (dstFrameId, srcFrameId, dstX, dstY, srcX, srcY, width, height))


os.system("mkdir -p outImg")
os.system("rm outImg/*")

imgBuffer = np.zeros((64, 720, 1280))


def movBlock(resImg, srcImg, dstX, dstY, srcX, srcY, width, height):
    if srcX < 0:
        width += srcX
        dstX -= srcX
        srcX = 0
    elif srcX + width > 1280:
        width -= (srcX + width - 1280)

    if srcY < 0:
        height += srcY
        dstY -= srcY
        srcY = 0
    elif srcY + height > 720:
        height -= (srcY + height - 720)
    # print(dstX, dstY, srcX, srcY, width, height)
    if width > 0 and height > 0:
        # resImg[dstY:dstY+height,dstX:dstX+width] = 255
        resImg[dstY:dstY+height, dstX:dstX +
               width] = srcImg[srcY:srcY+height, srcX:srcX+width]


def recoverImg(frameId):
    resImg = np.zeros((720, 1280))
    for dstFrameId, srcFrameId, dstX, dstY, srcX, srcY, width, height in MVlist[frameId]:
        movBlock(resImg, imgBuffer[srcFrameId % 64],
                 dstX, dstY, srcX, srcY, width, height)
    return resImg


for frameId, frameType in IPlist:
    if frameType == -2:  # I,P
        imgBuffer[frameId % 64] = cv2.imread(
            join(videoDir, "%06d.JPEG" % frameId), 0)
    elif frameType == -1:  # B
        imgBuffer[frameId % 64] = recoverImg(frameId)
    cv2.imwrite("outImg/%04d.JPEG" % frameId, imgBuffer[frameId % 64])
# I/P
# dstFrameId, srcFrameId, dstX, dstY, srcX, srcY, width, height
