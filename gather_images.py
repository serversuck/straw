import cv2
import os
import sys

try:
    label_name = sys.argv[1]
    num_samples = int(sys.argv[2])
except:
    print("Arguments missing.")
    print("this for collect image press a: to capture ; example : python gather_images.py cat 10 ")
    exit(-1)

IMG_SAVE_PATH = 'traindata' #name of dataset folder
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH)
#
# try:
#     os.mkdir(IMG_SAVE_PATH)
# except FileExistsError:
#     pass
# try:
#     os.mkdir(IMG_CLASS_PATH)
# except FileExistsError:
#     print("{} directory already exists.".format(IMG_CLASS_PATH))
#     print("All images gathered will be saved along with existing items in this folder")

cap = cv2.VideoCapture(0)

start = False
count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame , 1)
    if not ret:
        continue

    if count == num_samples:
        break

    cv2.rectangle(frame, (100, 0), (400, 300), (255, 255, 255), 2)

    k = cv2.waitKey(10)
    if k == ord('a'):
        roi = frame[100:400, 0:300]
        #save_path = os.path.join(label_name, '{}.jpg'.format(count + 1))
        save_path = IMG_SAVE_PATH + label_name+'.'+ format(count+1) +'.jpg'
        cv2.imwrite(save_path, roi)
        count += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    

    if k == ord('q'):
        break

print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()
