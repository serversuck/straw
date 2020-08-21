from numpy import loadtxt
import cv2, imutils
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle# load model
model = pickle.load(open('rf_model.sav', 'rb'))


def extract_color_histogram(image, bins=(8, 8, 8)):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    #frame = cv2.resize(frame , (640,400))
    frame = cv2.flip(frame ,1)
    
    roi = frame[100:400, 0:300]
    cv2.rectangle(frame, (100, 0), (400, 300), (255, 255, 255), 2)
    
    d = extract_color_histogram(roi)
    dnp = np.array(d)
    dnp = np.reshape(dnp, (1, 512) )
    # scale = StandardScaler()
#     x = scale.fit_transform(dnp)
    value = model.predict(dnp)[0].flatten()
    
    #print(value)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'result='+str(value[0]), (100,400), font, 1, (0,0,255),2)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
