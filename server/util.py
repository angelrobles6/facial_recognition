import joblib
import cv2
import json
import numpy as np
import base64
from wavelet import w2d


def classify_image(image_b64_data, file_path=None):
    images = get_cropped_image_if_2_eyes(file_path, image_b64_data)
    raw_image_val = 32 * 32 * 3
    har_img_val = 32 * 32

    for img in images:
        scaled_raw_img = cv2.resize(img, (32,32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har= cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(raw_image_val, 1), scaled_img_har(har_img_val, 1)))

        len_img_array = raw_image_val + har_img_val

        final = combined_img.reshape(1, len_img_array).astype(float)


def get_cv2_image_from_b64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_2_eyes(image_path, image_b64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_b64_string(image_b64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces


def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()


if __name__ == "__main__":
    print(classify_image(get_b64_test_image_for_virat(), None))