# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:55:40 2024

@author: Bryso
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


#Get the model weights and load in the object detection model
WeightsPath = "ModelWeights.pt"
model = torch.hub.load('YoloModel', 'custom', path= WeightsPath, source='local') 



#get the Image file
JPGImage = 'images/_ZED2i_2023-03-23-10_38_09-370777_png.rf.e92d4698f76813896744e8d3d18283a4.jpg'                  
                  
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