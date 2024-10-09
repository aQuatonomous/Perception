# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:55:40 2024

@author: Bryso
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

WeightsPath = "Best.pt"

test1 = '_ZED2i_2023-03-23-10_38_09-370777_png.rf.e92d4698f76813896744e8d3d18283a4.jpg'


model = torch.hub.load('yolov5', 'custom', path= WeightsPath, source='local') 
                       
                       
                       
img = cv2.imread(test1) 

plt.imshow(img)
plt.show()

results = model(img)

# Results
results.print()  
results.show()  # or .show()



print(results.pandas().xyxy[0])