# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import face_recognition, argparse
from aux_functions import *

# Command-line input setup
parser = argparse.ArgumentParser(
    description="MaskTheFace - Python code to mask faces dataset"
)
parser.add_argument(
    "--path",
    type=str,
    default="data1",
    help="Path to the folder containing images within folders as classes",
)
parser.add_argument(
    "--mask_type",
    type=str,
    default="all",
    help="Type of the mask to be applied. Available options: all, surgical_blue, surgical_green, N95, cloth",
)
# parser.add_argument(
#     "--verbose",
#     type=bool,
#     default=False,
#     help="Turn verbosity on/off. True or False",
# )
parser.add_argument('--verbose', dest='verbose', action='store_true', help="Turn verbosity on")
parser.set_defaults(feature=False)

args = parser.parse_args()
args.write_path = args.path + "_masked"
# Check in path is file or directory or none
is_directory, is_file, is_other = check_path(args.path)
display_MaskTheFace()
if is_directory:
    path, dirs, files = os.walk(args.path).__next__()
    file_count = len(files)
    dirs_count = len(dirs)

    # Deal files first
    # tqdm.write("Masking image files")
    print_orderly("Masking image files", 60)
    for f in tqdm(files):
        image_path = path + "/" + f

        write_path = path + "_masked"
        if not os.path.isdir(write_path):
            os.makedirs(write_path)

        if is_image(image_path):
            # Proceed if file is image
            if args.verbose:
                str_p = 'Processing: ' + image_path
                tqdm.write(str_p)

            split_path = f.rsplit(".")
            image, mask = mask_image(image_path, args.mask_type, args.verbose)
            for i in range(len(mask)):
                w_path = (
                    write_path
                    + "/"
                    + split_path[0]
                    + "_"
                    + mask[i]
                    + "."
                    + split_path[1]
                )
                img = image[i]
                cv2.imwrite(w_path, img)

    print_orderly("Masking image directories", 60)
    for d in tqdm(dirs):
        dir_path = args.path + "/" + d
        dir_write_path = args.write_path + "/" + d
        if not os.path.isdir(dir_write_path):
            os.makedirs(dir_write_path)
        _, _, files = os.walk(dir_path).__next__()
        for f in files:
            image_path = dir_path + "/" + f
            if args.verbose:
                str_p = 'Processing: ' + image_path
                tqdm.write(str_p)
            write_path = dir_write_path
            if is_image(image_path):
                # Proceed if file is image
                split_path = f.rsplit(".")
                image, mask = mask_image(image_path, args.mask_type, args.verbose)
                for i in range(len(mask)):
                    w_path = (
                        write_path
                        + "/"
                        + split_path[0]
                        + "_"
                        + mask[i]
                        + "."
                        + split_path[1]
                    )
                    img = image[i]
                    cv2.imwrite(w_path, img)
elif is_file:
    print("Masking image file")
    image_path = args.path
    write_path = args.path.rsplit(".")[0]
    if is_image(image_path):
        # Proceed if file is image
        image, mask = mask_image(image_path, args.mask_type, args.verbose)
        for i in range(len(mask)):
            w_path = write_path + "_" + mask[i] + "." + args.path.rsplit(".")[1]
            img = image[i]
            cv2.imwrite(w_path, img)
else:
    print('Path is neither a valid file or a valid directory')
print('Processing Done')
