# Author: aqeelanwar
# Created: 6 July,2020, 12:14 AM
# Email: aqeel.anwar@gatech.edu

from PIL import ImageColor
import cv2
import numpy as np

COLOR = [
    "#fc1c1a",
    "#177ABC",
    "#94B6D2",
    "#A5AB81",
    "#DD8047",
    "#6b425e",
    "#e26d5a",
    "#c92c48",
    "#6a506d",
    "#ffc900",
    "#ffffff",
    "#000000",
    "#49ff00",
]


def color_the_mask(mask_image, color, intensity):
    assert 0 <= intensity <= 1, "intensity should be between 0 and 1"
    RGB_color = ImageColor.getcolor(color, "RGB")
    RGB_color = (RGB_color[2], RGB_color[1], RGB_color[0])
    orig_shape = mask_image.shape
    bit_mask = mask_image[:, :, 3]
    mask_image = mask_image[:, :, 0:3]

    color_image = np.full(mask_image.shape, RGB_color, np.uint8)
    mask_color = cv2.addWeighted(mask_image, 1 - intensity, color_image, intensity, 0)
    mask_color = cv2.bitwise_and(mask_color, mask_color, mask=bit_mask)
    colored_mask = np.zeros(orig_shape, dtype=np.uint8)
    colored_mask[:, :, 0:3] = mask_color
    colored_mask[:, :, 3] = bit_mask
    return colored_mask


def texture_the_mask(mask_image, texture_path, intensity):
    assert 0 <= intensity <= 1, "intensity should be between 0 and 1"
    orig_shape = mask_image.shape
    bit_mask = mask_image[:, :, 3]
    mask_image = mask_image[:, :, 0:3]
    texture_image = cv2.imread(texture_path)
    texture_image = cv2.resize(texture_image, (orig_shape[1], orig_shape[0]))

    mask_texture = cv2.addWeighted(
        mask_image, 1 - intensity, texture_image, intensity, 0
    )
    mask_texture = cv2.bitwise_and(mask_texture, mask_texture, mask=bit_mask)
    textured_mask = np.zeros(orig_shape, dtype=np.uint8)
    textured_mask[:, :, 0:3] = mask_texture
    textured_mask[:, :, 3] = bit_mask

    return textured_mask



# cloth_mask = cv2.imread("masks/templates/cloth.png", cv2.IMREAD_UNCHANGED)
# # cloth_mask = color_the_mask(cloth_mask, color=COLOR[0], intensity=0.5)
# path = "masks/textures"
# path, dir, files = os.walk(path).__next__()
# first_frame = True
# col_limit = 6
# i = 0
# # img_concat_row=[]
# img_concat = []
# # for f in files:
# #     if "._" not in f:
# #         print(f)
# #         i += 1
# #         texture_image = cv2.imread(os.path.join(path, f))
# #         m = texture_the_mask(cloth_mask, texture_image, intensity=0.5)
# #         if first_frame:
# #             img_concat_row = m
# #             first_frame = False
# #         else:
# #             img_concat_row = cv2.hconcat((img_concat_row, m))
# #
# #             if i % col_limit == 0:
# #                 if len(img_concat) > 0:
# #                     img_concat = cv2.vconcat((img_concat, img_concat_row))
# #                 else:
# #                     img_concat = img_concat_row
# #                 first_frame = True
#
# ## COlor the mask
# thresholds = np.arange(0.1,0.9,0.05)
# for intensity in thresholds:
#     c=COLOR[2]
#     # intensity = 0.5
#     if "._" not in c:
#         print(intensity)
#         i += 1
#         # texture_image = cv2.imread(os.path.join(path, f))
#         m = color_the_mask(cloth_mask, c, intensity=intensity)
#         if first_frame:
#             img_concat_row = m
#             first_frame = False
#         else:
#             img_concat_row = cv2.hconcat((img_concat_row, m))
#
#             if i % col_limit == 0:
#                 if len(img_concat) > 0:
#                     img_concat = cv2.vconcat((img_concat, img_concat_row))
#                 else:
#                     img_concat = img_concat_row
#                 first_frame = True
#
#
# cv2.imshow("k", img_concat)
# cv2.imwrite("combine_N95_left.png", img_concat)
# cv2.waitKey(0)
# cc = 1
