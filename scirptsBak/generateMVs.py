import cv2
import numpy as np
import os
from os.path import join

baseDir = "1P3B_nodepend"

for video in sorted(os.listdir("ILSVRC2015/Data/VID/val/")):
    videoDir = "ILSVRC2015/Data/VID/val/%s" % video

    assert(os.path.exists(videoDir))
    imgShape = cv2.imread(join(videoDir, "000000.JPEG")).shape
    outDir = "%s/%s" % (baseDir, video)
    os.system("mkdir -p %s" % outDir)
    with open(join(outDir, "imageShape.txt"), "w") as f:
        f.write("%d,%d" % (imgShape[0], imgShape[1]))

    os.system("./FFmpeg/ffmpeg -i %s/%%06d.JPEG -c:v libx265 -x265-params 'bframes=3:b-adapt=0:b-pyramid=0' tmp.mkv" % videoDir)
    os.system("./FFmpeg/ffmpeg -i tmp.mkv tmp.mp4")
    # exit(1)
    os.system("rm tmp.m*")
    os.system("mv IPlist.txt %s/" % outDir)
    os.system("mv MotionVector.txt %s/" % outDir)