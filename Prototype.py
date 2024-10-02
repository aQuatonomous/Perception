import cv2 as opencv
import numpy as nppy
import rospy
import BuoyDetection.msg

def talker():
  rate = rospy.Rate(10) # 10hz
  
  # Cluster Integration is the name of where we are publishing
  # BouyDetection is the custom data structure that will be used to export the data
  # queue_size is the size of the buffer from messages stored at a time
  pub = rospy.Publisher('Cluster Integration', BuoyDetection, queue_size=60)

  # Object Detection is the name of the node
  # anonymous determines if anyone can subscribe to the node
  rospy.init_node('Object Detection', anonymous=False)

  while rospy.is_shutdown():
  vidCap = opencv.VideoCapture(0, opencv.CAP_DSHOW)

  if not vidCap.isOpened():
      print("Failed to open live feed")
      exit()

  while True:
      isOpen, frame = vidCap.read()
      if not isOpen:
        print("Failed to recieve frame")
        break

      hsv = opencv.cvtColor(frame, opencv.COLOR_BGR2HSV)
      lower_range = np.array([20, 100, 20])
      upper_range = np.array([35, 255, 255])
      mask = opencv.inRange(hsv, lower_range, upper_range)
      contours, hierarchy = opencv.findContours(mask, opencv.RETR_EXTERNAL, opencv.CHAIN_APPROX_SIMPLE)
      if len(contours) != 0:
        for contour in contours:
            if opencv.contourArea(contour) > 250:
                x, y, w, h = opencv.boundingRect(contour)
                opencv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
    opencv.imshow("Result", frame)
    if opencv.waitKey(1) == ord('q'):
        break
vidCap.release()
opencv.destroyAllWindows()
