# Author: aqeelanwar
# Created: 2 May,2020, 3:24 AM
# Email: aqeel.anwar@gatech.edu


import os, os.path
from shutil import copyfile

# simple version for working with CWD
# print([name for name in os.listdir('.') if os.path.isfile(name)])
count = 0
path = "/Users/aqeelanwar/PycharmProjects/maskface/lfw"

path, dirs, files = os.walk(path).__next__()
file_count = len(files)
threshold = 40

new_path = "/Users/aqeelanwar/PycharmProjects/maskface/lfw_" + str(threshold)
if not os.path.isdir(new_path):
    os.mkdir(new_path)
for i in range(len(dirs)):
    sub_path = path + "/" + dirs[i]
    _, subdirs, files = os.walk(sub_path).__next__()
    if len(files) >= threshold:
        count += 1
        for sub_file in files:
            src = sub_path + "/" + sub_file
            dst_dir = new_path + "/" + dirs[i]
            dst = dst_dir + "/" + sub_file
            if not os.path.isdir(dst_dir):
                os.mkdir(dst_dir)
            copyfile(src, dst)
            cc = 1

    # print_str = dirs[i]+': '+str(len(files))
    # print(print_str)
    # print(len(dirs))
print("count: ", count)
