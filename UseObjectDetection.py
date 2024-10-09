# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:55:40 2024

@author: Bryso
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

#import rospy
#import BuoyDetection.msg




#Get the model weights and load in the object detection model
WeightsPath = "ModelWeights.pt"
model = torch.hub.load('YoloModel', 'custom', path= WeightsPath, source='local') 






#get the Image file
JPGImage = 'images/_ZED2i_2023-03-23-10_41_13-750677_png.rf.524ff861d544a094ec098d4576420f1e.jpg'                  
                  
#Open the image     
img = cv2.imread(JPGImage) 


#Show the image
plt.imshow(img)
plt.show()

#run the model on the image
results = model(img)

# Results of the model
results.print()  
results.show()  # or .show()



print(results.pandas().xyxy[0]) #get data from results