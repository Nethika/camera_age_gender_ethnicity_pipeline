# Face Recognition, Age, Gender and Ethnicity Pipeline

This code uses dlib's face detector model, face landmark calculation model and dlib's face recognition tool. 
To predic the gender and age, a Keras ResNet model is used. 
To predict etyhnicity, a muclticlass VGG-16 convolution model (Keras) is used.


## Requirements:
```
Keras==2.1.5
imutils==0.4.5
numpy==1.13.3
dlib==19.8.1
scikit-image==0.13.1
scikit-learn==0.19.0
scipy==1.0.0
opencv-python==3.4.0+contrib

```
## To Run Codes:

```sh run/debug_pipeline.sh```

```run/camera_access.py``` saves images to the directory ```model_start```

```run/faceid_age_gender_ethnicity.py``` writes the face IDs,age, gender and ethnicity into the file: ```hashes/faces_json.json```

## Docker:
```docker rm --force pipeline```

```docker run --name pipeline -ti -v c:/Users/NethikaSuraweera/Documents/camera_age_gender_ethnicity_pipeline:/srv/run r_image /bin/bash```

