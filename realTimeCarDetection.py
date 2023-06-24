import cv2
import time
import numpy as np
from keras.models import load_model

class_names = ['notvehicle', 'Vehicle']

model = load_model('./carModel.h5')

car_classifier = cv2.CascadeClassifier('./storage.googleapis.com_kagglesdsdata_datasets_1954729_3223031_Required Files_cars.xml_X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com2F202306222.xml')

cap = cv2.VideoCapture('Cars Moving On Road Stock Footage - Free Download.mp4')

while cap.isOpened():
    # time.sleep(.05)

    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    cars = car_classifier.detectMultiScale(frame, 1.4, 2)


    for x,y,w,h in cars:
        sub_car_img = frame[y : y + h, x : x + w]
        resized = cv2.resize(sub_car_img, (224, 224))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 224, 224, 3))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        print(label)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, class_names[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('cars', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
