import cv2
import os
import sys
files = os.listdir(sys.argv[1])
size = (int(sys.argv[2]),int(sys.argv[2]))
for file_ in files:
    imgpath = os.path.join(sys.argv[1],file_)
    print imgpath
    img = cv2.imread(imgpath)
    img = cv2.resize(img,size,interpolation=cv2.INTER_NEAREST)
    print "write image path=",imgpath
    cv2.imwrite(imgpath,img)
