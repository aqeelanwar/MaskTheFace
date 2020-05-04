# Author: Aqeel Anwar(ICSRL)
# Created: 5/2/2020, 11:15 PM
# Email: aqeel.anwar@gatech.edu

from tqdm import tqdm
import random, os, cv2
import numpy as np

def get_dataset(path, img_size, seed=1):
    print('Converting images to dataset')
    path, dirs, files = os.walk(path).__next__()
    data = []
    num_classes = len(dirs)
    for i in tqdm(range(num_classes)):
        # Read Image
        # Resize
        # Append
        label = i
        label_name = dirs[i]
        sub_path = path + "/" + dirs[i]
        _, subdirs, files = os.walk(sub_path).__next__()
        for f in files:
            src = sub_path + "/" + f
            img = cv2.imread(src)
            img = cv2.resize(img, (img_size, img_size))
            data.append((img, label, label_name))

    #Shuffle the data
    random.Random(4).shuffle(data)
    return data, num_classes


def split_dataset(data, train_ratio=0.8):
    random.Random(4).shuffle(data)
    ind = int(len(data)*train_ratio)
    print('Train Images: ', ind)
    print('Test Images : ', len(data)-ind)
    train_split = data[0:ind]
    test_split = data[ind:]
    X_train, Y_train ,Y_train_label,X_test,Y_test,Y_test_label = [],[],[],[],[],[]
    for tuple in train_split:
        X_train.append(tuple[0])
        Y_train.append(tuple[1])
        Y_train_label.append(tuple[2])

    for tuple in test_split:
        X_test.append(tuple[0])
        Y_test.append(tuple[1])
        Y_test_label.append(tuple[2])

    return np.asarray(X_train), np.asarray([Y_train]).T, np.asarray([Y_train_label]).T, np.asarray(X_test), np.asarray([Y_test]).T, np.asarray([Y_test_label]).T

def analyze_dataset(data, num_classes):
    label = 0
    count=np.zeros(num_classes, dtype=int)
    print('Total Images: ', len(data))
    for d in data:
        count[d[1]]+=1

    for n in range(num_classes):
        print('Class {:2d}: {:4d}'.format(n, count[n]))

    print('Images per class:  {:3.1f}'.format(np.mean(count)))



def get_vggface2(path, img_size, num_classes,seed=1):
    print('Converting images to dataset')
    path, dirs, files = os.walk(path).__next__()
    data = []
    dir_array = []
    # num_classes = len(dirs)
    for i in tqdm(range(num_classes)):
        dir_array.append(dirs[i])
        # Read Image
        # Resize
        # Append
        label = i
        # label_name = dirs[i]
        label_name = ''
        sub_path = path + "/" + dirs[i]
        _, subdirs, files = os.walk(sub_path).__next__()
        for f in files:
            src = sub_path + "/" + f
            img = cv2.imread(src)
            img = cv2.resize(img, (img_size, img_size))
            data.append((img, label, label_name))

    #Shuffle the data
    # random.Random(4).shuffle(data)
    return data, num_classes


def shuffle_dataset(data):
    random.shuffle(data)
    cc=1