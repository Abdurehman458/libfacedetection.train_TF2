# this script is used to save name of all images in a txt file of results folder
# obtained from repo builtin test.py script. 
# This text file will be used to copy selected xml files from annotation folder

import os
import tqdm
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser(description='Textfile containing image names')
parser.add_argument('--type', default='tf', type=str,
                     help='create textfile of either pytorch or tensorflow results')
parser.add_argument('--path', default='/home/arm/Downloads/liface/libfacedetection.train_TF2', type=str,
                     help='create textfile of either pytorch or tensorflow results')

args = parser.parse_args()
if args.type == "tf":
    typee="results_tf"
else:
    typee="results_torch"
dirList=sorted(os.listdir(typee))
with open("test.txt", "w") as f:
    for file in dirList:
        f.write(os.path.splitext(file)[0]+'\n') 
    f.close()        

# This script just reads img names from txt file and copies the corresponding 
# Ground Truth xml files to a new folder
path="/home/arm/Downloads/liface/libfacedetection.train_TF2/data/WIDER_FACE_rect/annotations"  #folder containing all train/val/test annotations
dirList=os.listdir(path)

f = open("test.txt", "r")
f = f.read().splitlines()
for x in f:
    print(os.path.join(path,x+".xml"))
    
    try:
        copyfile(os.path.join(path,x+".xml"), os.path.join("/home/arm/Downloads/liface/libfacedetection.train_TF2/mAP/input/ground-truth", x+".xml"))
    except:
        print("Not found!!!!!",os.path.join(path,x+".xml"))

# This script just copies the detection results files to 
# Ground Truth xml files to mAP/input/detection-results
path="/home/arm/Downloads/liface/libfacedetection.train_TF2/mAP/input/ground-truth/"  #folder containing all train/val/test annotations
dirList=os.listdir(path)

print(len(dirList))
for x in dirList:
    print(x)
    # p=os.path.join(args.path,"mAP/"+typee,os.path.splitext(x)[0]+".txt")
    # print(p)
    # exit()
#     print(os.path.splitext(x)[0])
    # img_name = os.path.splitext(x)[0]
    try:
        copyfile(os.path.join(args.path,"mAP/"+typee,os.path.splitext(x)[0]+".txt"), os.path.join(args.path,"mAP/input/detection-results",os.path.splitext(x)[0]+".txt"))
    except:
        print("Not found!!!!!",os.path.join(args.path,"mAP/"+typee,os.path.splitext(x)[0]+".txt"))