import cv2
import schedule
import time
import sys
import os
import io
from PIL import Image                                                                                


# Camera 1 is the forward facing integrated web cam
camera_port = 0

#Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 10
img_index = 100001
camera = cv2.VideoCapture(camera_port)

root_path = 'C:\\Users\\NethikaSuraweera\\Documents\\TinMan_Git\\neural-networks\\camera_age_gender_ethnicity_pipeline'

#root_path = 'C:\\Users\\NethikaSuraweera\\OneDrive - Tinman Kinetics, LLC\\Nethika Backup\\21_people_recognition_pipeline'
# Captures a single image from the camera and returns it in PIL format
def get_image():
   global img_index
   global camera
   # read is the easiest way to get a full image out of a VideoCapture object.
   retval, im = camera.read()
   fimage = root_path + "\\model_start\\" + str(img_index) + ".jpg"
   print(fimage)
   img_index += 1
   cv2.imwrite(fimage, im)

   #img = Image.open(fimage)
   #img.show() 

   
   window_width = 1920
   window_height = 1080
   cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
   cv2.resizeWindow('dst_rt', window_width, window_height)

   cv2.imshow('dst_rt', im)
   cv2.waitKey(1)
   

   return True

# Ramp the camera - these frames will be discarded and are only used to allow the camera
# to adjust light levels, if necessary
for i in range(ramp_frames):
   temp, toss = camera.read()

schedule.every(4).seconds.do(get_image)

while 1:
   schedule.run_pending()
   time.sleep(1)

del(camera)