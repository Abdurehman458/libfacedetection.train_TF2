# Training for libfacedetection in TensorFlow 2

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

It is the training program for [libfacedetection](https://github.com/ShiqiYu/libfacedetection). The source code is based on [FaceBoxes.PyTorch](https://github.com/sfzhang15/FaceBoxes.PyTorch) and [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch).

Visualization of our network architecture: [[netron]](https://netron.app/?url=https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/tasks/task1/onnx/YuFaceDetectNet.onnx).


### Contents
- [Installation](#installation)
- [Training](#training)
- [Detection](#detection)
- [Evaluation on WIDER Face](#evaluation-on-wider-face)


## Installation
1. Install [PyTorch](https://pytorch.org/) >= v1.0.0 following official instruction.

2. Clone this repository. We will call the cloned directory as `$TRAIN_ROOT`.
```Shell
git clone https://github.com/ShiqiYu/libfacedetection.train
```

3. Install dependencies.
```shell
pip install -r requirements.txt
```

_Note: Codes are based on Python 3+._

## Training
1. Download [WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html) dataset, place the images under this directory:
  ```Shell
  $TRAIN_ROOT/data/WIDER_FACE_rect/images
  ```
  and create a symbol link to this directory from  
  ```Shell
  $TRAIN_ROOT/data/WIDER_FACE_landmark/images
  ```
2. Train the model using WIDER FACE:
  ```Shell
  cd $TRAIN_ROOT/tf2
  python3 train.py
  ```

## Detection
1. Set video directory in inference320.py on line (45):
```Shell
cd $TRAIN_ROOT/tf2/
python3 inference320.py 
```

## Evaluation on WIDER Face
1. Enter the directory.
```shell
cd $TRAIN_ROOT/tasks/task1/
```

2. Create a symbolic link to WIDER Face. `$WIDERFACE` is the path to WIDER Face dataset, which contains `wider_face_split/`, `WIDER_val`, `images`(from WIDER_VAL) folder  etc. for example:
```shell
ln -s /path/to/widerface/ widerface
```

3. Perform evaluation. To reproduce the following performance, run on the default settings. Run `python test.py --help` for more options.
```shell
mkdir results
python test.py
```

4. Download and run the [official evaluation tools](http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip). ***NOTE***: Matlab required! OR skip this step if you dont have MATLAB and follow alternate step 5 to calculate mAP. 
```shell
# download
wget http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip
# extract
unzip eval_tools.zip
# run the offical evaluation script
cd eval_tools
vim wider_eval.m # modify line 10 and line 21 according to your case
matlab -nodesktop -nosplash -r "run wider_eval.m;quit;"
```

5. First copy tf2/results files in mAP/results_tf folder run these scripts in the given order:
```shell
# cd to mAP folder
# run gen_txt.py and select type pt for pytorch and tf for tensorflow results. Also add path to the repo.
python gen_txt.py --type pt --path /path/to/libfacedetection.train_TF2
# or
python gen_txt.py --type tf --path /path/to/libfacedetection.train_TF2
# Now change dir to scripts folder
cd mAP/scripts/extra
#run xml conversion script
python convert_gt_xml.py
cd /mAP
#run main.py script
python main.py
```
### Performance on WIDER Face (Val)
Run on default settings: scales=[1.], confidence_threshold=0.3:
```
mAP_tf2=0.659, mAP_torch=0.72
```

## Citation
Our paper, which introduces a novel loss named Extended IoU (EIoU), is coming out soon. We trained our model using the EIoU loss and obtained a performance boost, see [Performance on WIDER Face (Val)](#performance-on-wider-face-(val)) for details. Stay tune for the release of our paper!
