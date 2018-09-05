import numpy as np
import cv2

facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, img = cap.read()

    faces = facecascade.detectMultiScale(
        # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        img, scaleFactor=1.2, minNeighbors=5, minSize=(10, 10))

    stackedfaces = None
    for (x, y, w, h) in sorted(faces, key=lambda x: (x[0], x[1])):
        roi = img[y:y + h, x:x + w]
        faceresized = cv2.resize(roi, (256, int(256 / w * h)))

        stackedfaces = faceresized if stackedfaces is None else np.concatenate((stackedfaces, faceresized), axis=0)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if stackedfaces is not None:
        cv2.imshow("Faces", stackedfaces)
        cv2.moveWindow("Faces", 0, 0)

    cv2.imshow('Camera', img)
    cv2.moveWindow("Camera", 256, 0)

    key = cv2.waitKey(30) & 0xff

    if key == 27: # escape key
        break

cap.release()
cv2.destroyAllWindows()