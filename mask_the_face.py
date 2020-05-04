# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import face_recognition
from aux_functions import *

n = 1
debug = False
for i in range(n):
    # image_path = "data/images" + str(i + 1) + ".jpg"
    # image_write = "data/images" + str(i + 1) + "_masked.jpg"
    image_path = "data/group.jpg"
    image_write = "data/igroup_masks.jpg"

    # Get face landmarks
    image = face_recognition.load_image_file(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_landmarks_list = face_recognition.face_landmarks(image)
    face_locations = face_recognition.face_locations(image)
    # draw_landmarks(face_landmarks_list[0], image)

    # Process each face in the image
    for face_landmarks, face_location in zip(face_landmarks_list, face_locations):
        # Get key points
        six_points_on_face, angle = get_six_points(face_landmarks, image)
        plot_lines(six_points_on_face, image, debug=debug)
        # Put mask on face
        image = mask_face(
            image, face_location, six_points_on_face, angle, type="surgical"
        )
    out_img = image
    cv2.imshow(image_path, out_img)
    cv2.imwrite(image_write, out_img)

key = cv2.waitKey(0)
