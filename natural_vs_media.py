# Code for making ML classifiers' input vector
# Two classes for classifiers are 0 (media) and 1 (natural)
from joblib import load
from create_input_vector import create_feature_input
import warnings
warnings.filterwarnings('ignore')

#enter path to saved .wav file (record at 16khz on Kinect)
filename = 'replace this with path to file'

#load the scaler
scaler = load('scaler.save')

#extract audio features
feature_vector = create_feature_input(filename)

#scale the features
feature_vector = scaler.transform(feature_vector)

#load the respective classifier (example below)
svc = load('SVC.joblib')

#make the corresponding prediction using classifer.predict(feature_vector) -- example below
# Two classes for classifiers are 0 (media) and 1 (natural)
prediction = svc.predict(feature_vector)[0]

if(prediction==0):
    print("The audio file was predicted as media")
else:
    print("The audio file was predicted as natural")
