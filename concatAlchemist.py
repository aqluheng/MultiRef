import os
import torch
from os.path import join

import argparse
parser = argparse.ArgumentParser(description="Alchemist Result Concat")
parser.add_argument("--model", "-m",
                    help="The model which generate the detection results.",
                    required=True
                    )

parser.add_argument("--window", "-w",
                    help="The window for predict",
                    required=True,
                    type=int
                    )

args = parser.parse_args()
assert(args.model in ["DFF", "FGFA", "MEGA", "GT"])
assert(args.window in [1, 3, 7, 15])


baseDir = "AlchemistOut/base%s/1I%dP/" % (args.model, args.window)
allBBoxes = []
for filename in sorted(os.listdir(baseDir)):
    if filename[:10] != "ILSVRC2015":
        continue
    filepath = join(baseDir, filename)
    bboxList = torch.load(filepath)
    allBBoxes += bboxList

print(len(allBBoxes))
torch.save(allBBoxes, join(baseDir, "Alchemist1I%dP.pth"%args.window))
