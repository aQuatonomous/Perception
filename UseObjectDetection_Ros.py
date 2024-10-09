# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:55:40 2024

@author: Bryso
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from imutils.video import VideoStream
from imutils.video import FPS

import rospy
import BuoyDetection.msg
from std_msgs.msg import String
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Int32MultiArray 
from sensor_msgs.msg import Image



#Get the model weights and load in the object detection model
WeightsPath = "ModelWeights.pt"
model = torch.hub.load('YoloModel', 'custom', path= WeightsPath, source='local') 



def image_callback(msg):
    print("Received an image!")
    img = cv2.imread(msg) 
    results = model(img) #Run Object Detection Model
    
    pub.publish(results.pandas().xyxy[0]) #publish the results


def talker():
    rate = rospy.Rate(10) # 10hz
    # Cluster Integration is the name of where we are publishing
    # BouyDetection is the custom data structure that will be used to export the data
    # queue_size is the size of the buffer from messages stored at a time
    
    sub = rospy.Subscriber("Camera", Image, image_callback) #setup ros subscriber listening to Camera node and running Image_callback()
    # Object Detection is the name of the node
    # anonymous determines if anyone can subscribe to the node
    rospy.init_node('Object Detection', anonymous=False)
    

    
    
    
if __name__ == '__main__':
    pub = rospy.Publisher('Cluster Integration', BuoyDetection, queue_size=60)
    talker()
    
    