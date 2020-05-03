# Author: aqeelanwar
# Created: 27 April,2020, 10:21 PM
# Email: aqeel.anwar@gatech.edu

import numpy as np
import cv2, math, os, random
from PIL import Image, ImageDraw
from tqdm import tqdm


def get_line(face_landmark, image, type="eye"):
    debug = False
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    left_eye = face_landmark["left_eye"]
    right_eye = face_landmark["right_eye"]
    left_eye_mid = np.mean(np.array(left_eye), axis=0)
    right_eye_mid = np.mean(np.array(right_eye), axis=0)
    eye_line_mid = (left_eye_mid + right_eye_mid) / 2

    if type == "eye":
        left_point = left_eye_mid
        right_point = right_eye_mid
        mid_point = eye_line_mid

    elif type == "nose_mid":
        nose_length = (
            face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1]
        )
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length / 2]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length / 2]
        # mid_point = (
        #     face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
        # ) / 2

        mid_pointY = (
            face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
        ) / 2
        mid_pointX = (
            face_landmark["nose_bridge"][-1][0] + face_landmark["nose_bridge"][0][0]
        ) / 2
        mid_point = (mid_pointX, mid_pointY)

    elif type == "nose_tip":
        nose_length = (
            face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1]
        )
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length]
        mid_point = (
            face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
        ) / 2

    elif type == "bottom_lip":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.max(np.array(bottom_lip), axis=0)
        shiftY = bottom_lip_mid[1] - eye_line_mid[1]
        left_point = [left_eye_mid[0], left_eye_mid[1] + shiftY]
        right_point = [right_eye_mid[0], right_eye_mid[1] + shiftY]
        mid_point = bottom_lip_mid

    elif type == "perp_line":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.mean(np.array(bottom_lip), axis=0)

        left_point = eye_line_mid
        left_point = face_landmark["nose_bridge"][0]
        right_point = bottom_lip_mid

        mid_point = bottom_lip_mid

    elif type == "nose_long":
        nose_bridge = face_landmark["nose_bridge"]
        left_point = [nose_bridge[0][0], nose_bridge[0][1]]
        right_point = [nose_bridge[-1][0], nose_bridge[-1][1]]

        mid_point = left_point

    # d.line(eye_mid, width=5, fill='red')
    y = [left_point[1], right_point[1]]
    x = [left_point[0], right_point[0]]
    # cv2.imshow('h', image)
    # cv2.waitKey(0)
    eye_line = fit_line(x, y, image)
    d.line(eye_line, width=5, fill="blue")

    # Perpendicular Line
    # (midX, midY) and (midX - y2 + y1, midY + x2 - x1)
    y = [
        (left_point[1] + right_point[1]) / 2,
        (left_point[1] + right_point[1]) / 2 + right_point[0] - left_point[0],
    ]
    x = [
        (left_point[0] + right_point[0]) / 2,
        (left_point[0] + right_point[0]) / 2 - right_point[1] + left_point[1],
    ]
    perp_line = fit_line(x, y, image)
    if debug:
        d.line(perp_line, width=5, fill="red")
        pil_image.show()
    return eye_line, perp_line, left_point, right_point, mid_point


def get_points_on_chin(line, face_landmark):
    chin = face_landmark["chin"]
    points_on_chin = []
    for i in range(len(chin) - 1):
        chin_first_point = [chin[i][0], chin[i][1]]
        chin_second_point = [chin[i + 1][0], chin[i + 1][1]]

        flag, x, y = line_intersection(line, (chin_first_point, chin_second_point))
        if flag:
            points_on_chin.append((x, y))

    cc = 1
    return points_on_chin


def plot_lines(face_line, image):
    debug = False
    pil_image = Image.fromarray(image)
    if debug:
        d = ImageDraw.Draw(pil_image)
        d.line(face_line, width=4, fill="white")
        pil_image.show()


def line_intersection(line1, line2):
    mid = int(len(line1) / 2)
    # start = int(mid - mid / 5)
    # end = int(mid + mid / 5)
    start = 0
    end = -1
    line1 = ([line1[start][0], line1[start][1]], [line1[end][0], line1[end][1]])

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    x = []
    y = []
    flag = False

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return flag, x, y

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    segment_minX = min(line2[0][0], line2[1][0])
    segment_maxX = max(line2[0][0], line2[1][0])

    segment_minY = min(line2[0][1], line2[1][1])
    segment_maxY = max(line2[0][1], line2[1][1])

    if (
        x <= segment_maxX
        and x >= segment_minX
        and y <= segment_maxY
        and y >= segment_minY
    ):
        flag = True

    return flag, x, y


def get_closest_point(line, chin):
    point_on_chin = []
    for i in range(len(chin) - 1):
        chin_first_point = [chin[i][0], chin[i][1]]
        chin_second_point = [chin[i + 1][0], chin[i + 1][1]]
    cc = 1


line = ((0, 100), (100, 100))


def fit_line(x, y, image):
    if x[0] == x[1]:
        x[0] += 0.1
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(0, image.shape[1], 500)
    y_axis = polynomial(x_axis)
    eye_line = []
    for i in range(len(x_axis)):
        eye_line.append((x_axis[i], y_axis[i]))

    return eye_line


def get_six_points(face_landmark, image):
    _, _, _, _, m = get_line(face_landmark, image, type="nose_mid")
    face_b = m

    perp_line, _, _, _, _ = get_line(face_landmark, image, type="perp_line")
    points = get_points_on_chin(perp_line, face_landmark)
    face_e = points[0]

    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="nose_long")

    angle = get_angle(perp_line, nose_mid_line)
    print("angle: ", angle)
    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="nose_tip")
    points = get_points_on_chin(nose_mid_line, face_landmark)
    face_a = points[0]
    face_c = points[1]

    nose_mid_line, _, _, _, _ = get_line(face_landmark, image, type="bottom_lip")
    points = get_points_on_chin(nose_mid_line, face_landmark)
    face_d = points[0]
    face_f = points[1]

    six_points = np.float32([face_a, face_b, face_c, face_f, face_e, face_d])

    return six_points, angle


def get_angle(line1, line2):
    delta_y = line1[-1][1] - line1[0][1]
    delta_x = line1[-1][0] - line1[0][0]
    perp_angle = math.degrees(math.atan2(delta_y, delta_x))
    if delta_x < 0:
        perp_angle = perp_angle + 180
    if perp_angle < 0:
        perp_angle += 360
    if perp_angle > 180:
        perp_angle -= 180

    # print("perp", perp_angle)
    delta_y = line2[-1][1] - line2[0][1]
    delta_x = line2[-1][0] - line2[0][0]
    nose_angle = math.degrees(math.atan2(delta_y, delta_x))

    if delta_x < 0:
        nose_angle = nose_angle + 180
    if nose_angle < 0:
        nose_angle += 360
    if nose_angle > 180:
        nose_angle -= 180
    # print("nose", nose_angle)

    angle = nose_angle - perp_angle
    return angle


def mask_face(image, six_points, angle, T, type="surgical"):
    debug = True
    threshold = 13

    if type == "N95":
        if angle < threshold:
            mask_a = (62, 111)
            mask_b = (196, 21)
            mask_c = (619, 120)
            mask_d = (101, 434)
            mask_e = (217, 541)
            mask_f = (607, 398)
            img = cv2.imread("masks/N95_left.png")
        elif angle >= threshold:
            mask_a = (62, 111)
            mask_b = (196, 21)
            mask_c = (649, 120)
            mask_d = (101, 434)
            mask_e = (217, 411)
            mask_f = (647, 398)
            img = cv2.imread("masks/N95_left.png")
        else:

            mask_a = (32, 110)
            mask_b = (401, 21)
            mask_c = (758, 98)
            mask_d = (129, 498)
            mask_e = (392, 639)
            mask_f = (675, 490)
            img = cv2.imread("masks/N95.png")

    elif type == "surgical":
        if angle > threshold:
            mask_a = (39, 27)
            mask_b = (118, 9)
            mask_c = (488, 20)
            mask_d = (44, 267)
            mask_e = (168, 282)
            mask_f = (487, 202)
            img = cv2.imread("masks/surgical_mask_left.png")
        elif angle < -threshold:
            # Edit this
            mask_a = (28, 20)
            mask_b = (375, 9)
            mask_c = (466, 27)
            mask_d = (27, 202)
            mask_e = (337, 282)
            mask_f = (418, 267)
            img = cv2.imread("masks/surgical_mask_right.png")
        else:
            mask_a = (41, 97)
            mask_b = (307, 22)
            mask_c = (570, 99)
            mask_d = (55, 322)
            mask_e = (295, 470)
            mask_f = (555, 323)
            img = cv2.imread("masks/surgical_mask.png")

    # Change brightness
    img = cv2.add(img, np.array([(T - 255) / 3]))
    w = image.shape[0]
    h = image.shape[1]
    mask_line = np.float32([mask_a, mask_b, mask_c, mask_f, mask_e, mask_d])

    # M = cv2.getPerspectiveTransform(mask_line, face_line)
    M, mask = cv2.findHomography(mask_line, six_points)
    matchesMask = mask.ravel().tolist()
    # print(matchesMask)
    dst_mask = cv2.warpPerspective(img, M, (h, w))

    dst_mask_points = cv2.perspectiveTransform(mask_line.reshape(-1, 1, 2), M)

    img2gray = cv2.cvtColor(dst_mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('mask_mask', mask)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(image, image, mask=mask)

    img2_fg = cv2.bitwise_and(dst_mask, dst_mask, mask=mask_inv)
    # cv2.imshow('f', img2_fg)
    # cv2.waitKey(0)
    img1_bg = cv2.cvtColor(img1_bg, cv2.COLOR_BGR2RGB)
    out_img = cv2.add(img1_bg, img2_fg)

    if debug:
        for i in six_points:
            cv2.circle(out_img, (i[0], i[1]), radius=4, color=(0, 0, 255), thickness=-1)

        for i in dst_mask_points:
            cv2.circle(
                out_img, (i[0][0], i[0][1]), radius=4, color=(0, 255, 0), thickness=-1
            )

    # cv2.imshow("i", out_img)
    # cv2.waitKey(0)

    return out_img


def draw_landmarks(face_landmarks, image):
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5, fill="white")

    pil_image.show()



