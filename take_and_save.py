
import os
import re
import numpy as np
import cv2
from PIL import Image

#from matplotlib import pyplot, cm
import matplotlib.pyplot as plt
x='D:/research/RESULT_CHEST_CT_gamma_correction/RESULT_z/'
output = [dI for dI in os.listdir(x) if os.path.isdir(os.path.join(x,dI))]
output=output[0:len(output)]
print(output)


for i in range(len(output)):
    Path='D:/research/RESULT_CHEST_CT_gamma_correction/RESULT_z/{}/'.format(output[i])
    '''
    temp = re.findall(r'\d+', output[i])
    res = list(map(int, temp))
    strings = [str(integer) for integer in res]
    a_string = "".join(strings)
    an_integer = int(a_string)
    print(an_integer)
    '''
    h=output[i] #dest. folder name
    #print(h)



    lstFilesDCM = []  # create an empty list
    imagename=[]
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if "normal.png" in filename.lower():  # check whether the file's DICOM
                imagename.append(filename)
                lstFilesDCM.append(os.path.join(dirName, filename))
    #k=lstFilesDCM
    # loop through all the png files

    i=0
    for filepath in lstFilesDCM:
        #print(filepath)
        p=imagename[i]
        #print(p)
        p=p[:-4]
        #print(p)

        image = cv2.imread(filepath)


        # cv2.imshow("Images", adjusted)
        # cv2.waitKey(0)
        cv2.imwrite('D:/polynomial/UNET2D/data/xray/train_z/normal_{}.png'.format(h), image)

    #if (i==200):
     #   break
    #i=i+1