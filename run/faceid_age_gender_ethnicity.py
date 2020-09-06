import io
import os
import time
import math
import sys
#import watchdog
#from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

import dlib
import glob
from skimage import io as skiopen
from imutils.face_utils import FaceAligner
import imutils
from imutils import paths
import numpy as np
from scipy.spatial import distance
import json
import string
from keras.optimizers import SGD
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
import cv2
import tensorflow as tf
import pandas as pd

# Load Models
## Face landmarks model:
predictor_path = "/srv/rugby/models/shape_predictor_68_face_landmarks.dat" 
sp = dlib.shape_predictor(predictor_path)
## Face recognition model(# calculates 128D vector (hash) for an image):
face_rec_model_path = "/srv/rugby/models/dlib_face_recognition_resnet_model_v1.dat"
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
## model from dlib to detect faces:
detector = dlib.get_frontal_face_detector()
## to allign the face    
fa = FaceAligner(sp, desiredFaceWidth=512)  

## Age and Gender model
model_a_g = load_model("/srv/rugby/models/gender_age.model")
model_a_g._make_predict_function()
## Ethnicity model
model_ethnicity = load_model("/srv/rugby/models/ethnicity.model")
model_ethnicity._make_predict_function()
graph = tf.get_default_graph()

class Watcher:
    DIRECTORY_TO_WATCH = "/srv/rugby/model_start"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
            print("Watcher for FaceID Stopped")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        global sp
        global facerec
        global detector
        global fa
        global model_a_g
        global model_ethnicity
        global graph
        
        if event.is_directory:
            return None

        #elif event.event_type == 'created' and event.src_path[-4:] == ".jpg":
        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            print("Received created event - %s." % event.src_path)

            new_image_path = event.src_path


            print("new_image_path:")
            print(new_image_path)

            #img = skiopen.imread(new_image_path)
            img = cv2.imread(new_image_path)
            print("Image Loaded")
            # Delete image file
            if os.path.isfile(new_image_path):
                os.remove(new_image_path)
            
            # input_img: RGB  (input to face detector)
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(img)
            img_size_a_g = 64
            img_size_e = 100

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            print("Faces Detected:",len(detected))

            faces = np.empty((len(detected), img_size_a_g, img_size_a_g, 3))

            # Open the hashes
            csv_file = "/srv/rugby/hashes/faces_csv.csv"

            classes = ['african', 'asianindian', 'caucasian', 'eastasian', 'latino']
            ethnicity_class =[]

            features = []

            if len(detected) > 0:
                # Gender and Age
                for i, d in enumerate(detected):

                    # create face hash
                    shape_new = sp(input_img, d)     # face landmarks model
                    face_descriptor = facerec.compute_face_descriptor(input_img, shape_new)     # face recognition model
                    features.append(face_descriptor)

                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - 0.4 * w), 0)
                    yw1 = max(int(y1 - 0.4 * h), 0)
                    xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                    yw2 = min(int(y2 + 0.4 * h), img_h - 1)

                    faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size_a_g, img_size_a_g))

                    print("Calculating Ethnicity.....")
                    face_only = img[y1:y2,x1:x2]
                    face_gray = (cv2.cvtColor(face_only, cv2.COLOR_RGB2GRAY)*(1./255))
                    im = cv2.resize(face_gray, (img_size_e, img_size_e))
                    im = im.reshape((1,img_size_e,img_size_e,1))
                    with graph.as_default():
                        class_labels = model_ethnicity.predict_classes(im, verbose=0)[0]

                    print("Face #: ",i)
                    print ("Ethnicity:",classes[class_labels])
                    ethnicity_class.append(classes[class_labels])

                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                features = np.array(features)   


                print("Calculating Age and Gender.....")
                # predict ages and genders of the detected faces
                results = model_a_g.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                print("Predicted Ages:",predicted_ages)
                print("Predicted Gender:",predicted_genders)

                # Read from csv file
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, delimiter=',')
                    face_data = eval(df.to_json(orient='records'))
                else:
                    face_data =[]

                # Threshold set to identify different faces.  Might need tuning in case of siblings, sun glass pictures etc.  
                threshold=0.4

                #
                for i in range(len(features)): #faces in new image
                    inface_hash= features[i]
                    match_dict={}
                    print (i)
                    for j in range(len(face_data)):
                        face_id = face_data[j]['face_id']
                        face_freq = face_data[j]['frequency']
                        face_hash = json.loads(face_data[j]['hash'])
                        face_age = face_data[j]['face_age']
                        face_gender = face_data[j]['face_gender']
                        face_ethnicity = face_data[j]['face_ethnicity']
                        dist = distance.euclidean(inface_hash,face_hash)
                        #print(face_id , face_freq, dist)
                        if dist < threshold:
                            match_dict[j] = dist
                    if match_dict:
                        indx = min(match_dict, key=match_dict.get)
                        min_hash = json.loads(face_data[indx]['hash'])
                        # find New Mean for hash
                        new_mean = np.mean([inface_hash,min_hash],axis=0)
                        #update hash
                        face_data[indx]['hash'] = str(new_mean.tolist())
                        #update frequency
                        face_data[indx]['frequency'] += 1  
                        # update age
                        face_data[indx]['face_age'] = (face_data[indx]['face_age'] + predicted_ages[i] )/2.0
                        print("Matched with:")
                        print(face_data[indx]['face_id'])
                    else:     #new face
                        print("No match! -> New face:")
                        face_id = str(len(face_data)+1).zfill(4)
                        print(face_id)
                        tempt_dict={'face_id': face_id, 'frequency': 1,'face_age':predicted_ages[i],'face_gender':"F" if predicted_genders[i][0] > 0.5 else "M",'face_ethnicity':ethnicity_class[i], 'hash':str(inface_hash.tolist())}
                        face_data.append(tempt_dict)
                            
                #update csv file
                df = pd.DataFrame(face_data)
                df.to_csv(csv_file, index = False)
                    

        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            print("Received modified event - %s." % event.src_path)

        elif event.event_type == 'deleted':
            # Taken any action here when a file is modified.
            print("Recieved deleted event - %s." % event.src_path)

        else:
            print("Event recieved: %s." % event.event_type)


if __name__ == '__main__':
    w = Watcher()
    w.run()
