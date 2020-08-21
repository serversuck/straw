from os import listdir
import os
from os.path import isfile, join
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import cv2, imutils
import numpy as np
from sklearn.model_selection import  train_test_split

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()

#prepare file
mypath = 'traindata'
imagePaths = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]


rawImages=[]
labels=[]
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    pixels = extract_color_histogram(image)
    label = imagePath.split(os.path.sep)[-1].split('.')[0]

    rawImages.append(pixels)
    labels.append(label)

    if i > 0 and i % 2 == 0:
        print('[INFO] processed {} / {}'.format(i, len(imagePaths)))

x = np.array(rawImages)
y = np.array(labels)

#re-scale  x
# scale = StandardScaler()
# x = scale.fit_transform(x)

#split data
x_train, x_test, y_train, y_test = train_test_split(x , y ,train_size = 0.8 , random_state=42)

#import and create model
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

svm_model = svm.SVC(kernel='linear') 
rf_model =RandomForestClassifier(n_estimators=100)
knn_model = KNeighborsClassifier(n_neighbors = 5)
dt = tree.DecisionTreeClassifier()

svm_model = svm_model.fit(x_train, y_train)
rf_model = rf_model.fit(x_train, y_train)
knn_model= knn_model.fit(x_train, y_train)
dt = dt.fit(x_train, y_train)

s_score = svm_model.score(x_test , y_test)
r_score = rf_model.score(x_test , y_test)
k_score = knn_model.score(x_test , y_test)
d_score = dt.score(x_test , y_test)

print(s_score, r_score, k_score, d_score)

resultSVM = svm_model.predict(x_test)
resultRF = rf_model.predict(x_test)
resultKNN = knn_model.predict(x_test)
resultDT = dt.predict(x_test)
print('-----------------------------------------------------------')
print('ของจริง ', y_test)
print('-----------------------------------------------------------')
print('svn ทำนาย' , resultSVM)
print('rf ทำนาย' , resultRF)
print('knn ทำนาย' , resultKNN)
print('dt ทำนาย' , resultDT)

import pickle
filename = 'rf_model.sav'
pickle.dump(rf_model, open(filename, 'wb'))
print('-----------------------------------------------------------')
print('train finished and save random forrest model...')

