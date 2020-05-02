# Author: aqeelanwar
# Created: 2 May,2020, 5:27 AM
# Email: aqeel.anwar@gatech.edu
import os
import numpy as np
import urllib.request
import cv2


def url_to_image(url):
    image_downloaded = True
    try:
        resp = urllib.request.urlopen(url, timeout=1)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            image_downloaded = False
    except:
        image = []
        image_downloaded = False

    return image_downloaded, image


path = "files"
path, dirs, files = os.walk(path).__next__()
file_count = len(files)

abs_path = os.path.abspath(os.getcwd())
if not os.path.isdir(abs_path + "/subset"):
    os.mkdir(abs_path + "/subset")


for person in files[0:2]:
    write_file = abs_path + "/subset" + "/" + person.rsplit(".", 1)[0]
    if not os.path.isdir(write_file):
        os.mkdir(write_file)
    text_file = path + "/" + person
    f = open(text_file, "r")
    lines = f.readlines()
    i = 0
    for line in lines:
        l = line.split(" ")
        if len(l) == 9:
            # If the split was correct
            url = l[1]
            proceed, image = url_to_image(url)
            # print("received")
            if proceed:
                # [left top right bottom]
                left = int(float(l[2]))
                top = int(float(l[3]))
                right = int(float(l[4]))
                bottom = int(float(l[5]))

                width = right - left
                height = bottom - top

                # left -= int(width / 5)
                # right += int(width / 5)
                # top -= int(height / 5)
                # bottom += int(height / 5)

                # cv2.imshow("cropped", image)
                # cv2.waitKey(0)
                # cc = 1
                cropped_image = image[top:bottom, left:right]
                # cv2.imshow("cropped", cropped_image)
                # cv2.waitKey(1)

                if i == 45:
                    cc = 1

                if cropped_image.size != 0:
                    cropped_image = cv2.resize(cropped_image, (224, 224))
                    num = str(i).zfill(5)
                    write_filepath = (
                        write_file + "/" + num + ".png"
                    )
                    cv2.imwrite(write_filepath, cropped_image)

                    print(i)
                    i += 1

    # print(f.read())
