# Author: aqeelanwar
# Created: 2 May,2020, 2:49 AM
# Email: aqeel.anwar@gatech.edu

from tkinter import filedialog
from tkinter import *
import cv2, os

mouse_pts = []


def get_mouse_points(event, x, y, flags, param):
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(mask_im, (x, y), 10, (0, 255, 255), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        # print("Point detected")
        # print((x,y))


root = Tk()
filename = filedialog.askopenfilename(
    initialdir="/",
    title="Select file",
    filetypes=(("PNG files", "*.PNG"), ("png files", "*.png"), ("All files", "*.*")),
)
root.destroy()
filename_split = os.path.split(filename)
folder = filename_split[0]
file = filename_split[1]
file_split = file.split(".")
new_filename = folder + "/" + file_split[0] + "_marked." + file_split[-1]
mask_im = cv2.imread(filename)
cv2.namedWindow("Mask")
cv2.setMouseCallback("Mask", get_mouse_points)

while True:
    cv2.imshow("Mask", mask_im)
    cv2.waitKey(1)
    if len(mouse_pts) == 6:
        cv2.destroyWindow("Mask")
        break
    first_frame_display = False
points = mouse_pts
print(points)
print("----------------------------------------------------------------")
print("Copy the following code and paste it in masks.cfg")
print("----------------------------------------------------------------")
name_points = ["a", "b", "c", "d", "e", "f"]

mask_title = "[" + file_split[0] + "]"
print(mask_title)
print("template: ", filename)
for i in range(len(mouse_pts)):
    name = (
        "mask_"
        + name_points[i]
        + ": "
        + str(mouse_pts[i][0])
        + ","
        + str(mouse_pts[i][1])
    )
    print(name)

cv2.imwrite(new_filename, mask_im)
