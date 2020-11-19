import os
import dlib
import imagesize


def get_ann_rectangles(img_path, ann_path):
    """Returns a dlib.rectangles object depending on the image annotations.
    """
    img_width, img_height = imagesize.get(img_path)
    img_name = os.path.basename(img_path)
    name, _ = os.path.splitext(img_name)
    name = name + ".txt"
    ann_path = os.path.join(ann_path, name)

    annots = [x.rstrip('\n') for x in open(ann_path, 'r')]
    dlib_rectangles = dlib.rectangles()
    for line in annots:
        coords = line.split(' ')
        label_num = float(coords[0])

        norm_x_center = float(coords[1])
        norm_y_center = float(coords[2])
        norm_width = float(coords[3])
        norm_height = float(coords[4])
        left = int((norm_x_center - (norm_width * 0.5)) * img_width)
        top = int((norm_y_center - (norm_height * 0.5)) * img_height)
        width = norm_width * img_width
        height = norm_height * img_height
        right = int(left + width)
        bottom = int(top + height)

        dlib_rectangles.append(dlib.rectangle(left, top, right, bottom))
    return dlib_rectangles
