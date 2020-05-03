# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

from PIL import Image, ImageDraw
import face_recognition
from aux_functions import *
import numpy as np

n = 19
for i in range(n):
    image_path = 'data/images'+str(i+1)+'.jpg'
    image_write = 'data/images'+str(i+1)+'_masked.jpg'
    # image_path = "data/images1.jpg"
    # image_write = "data/images1_masked_new.jpg"
    # image_path = 'data/test.png'
    # image_write = 'data/test.png'
    print(image_path)
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    R = im[0]
    G = im[1]
    B = im[2]
    brightness_factor = np.mean(0.2126 * R + 0.7152 * G + 0.0722 * B)

    image = face_recognition.load_image_file(image_path)
    face_landmarks_list = face_recognition.face_landmarks(image)
    # draw_landmarks(face_landmarks_list[0], image)

    # get_angle(face_landmarks_list[0], image)
    six_points_on_face, angle = get_six_points(face_landmarks_list[0], image)
    plot_lines(six_points_on_face, image)

    out_img = mask_face(image, six_points_on_face, angle, brightness_factor, type="cloth")

    cv2.imshow(image_path, out_img)
    cv2.imwrite(image_write, out_img)

key = cv2.waitKey(0)
