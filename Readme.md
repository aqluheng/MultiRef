### 进入环境
conda activate MEGA

### 代码目录
- MultiRef/Alchemist_det.py 执行Alchemist算法,结果写入AlchemistOut
    - python Alchemist_det.py -m DFF -w 1 
- MultiRef/concatAlchemist.py 拼接所有单视频结果并生成.pth文件供准确率测试
    - python concatAlchemist.py -m DFF -w 1
- mega.pytorch/tools/test_prediction_VID.py 输入555个test视频的结果输出准确率 
    - python tools/test_prediction_VID.py -p ../MultiRef/AlchemistOut/baseDFF/1I1P/Alchemist1I3P.pth -ms



### 文件夹解释
- annoPth 包含DFF,FGFA,MEGA以及GT的检测框结果
- AlchemistOut 包含Alchmist的结果,包含多个路径如baseMEGA/1I1P/
    - ILSVRC...为单视频的检测结果
    - baseMEGA/1I1P/Alchemist1I1P.pth为合并结果
- videoInfo 包含复原所需的视频信息,以1I1P/ILSVRC...
    - imageShape.txt 包含图片大小
    - IPlist.txt 包含IPB帧编号
    - MotionVector.txt 包含所有运动矢量
- FFmpeg 用于生成运动矢量的源代码文件
- ILSVRC2015 原数据集
- scriptsBak 包含一些用不到的脚本
    - generateMVs.py 生成所有视频的videoInfo
    - testMV.py 测试运动矢量是否合规
    - recoverFrame.py 根据运动矢量还原图片
    - transformPrediction.py 将gt转换为gtPrediction.pt的格式