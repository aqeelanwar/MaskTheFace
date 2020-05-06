# Author: Aqeel Anwar(ICSRL)
# Created: 5/2/2020, 11:15 PM
# Email: aqeel.anwar@gatech.edu

from tqdm import tqdm
import random, os, cv2
import numpy as np
import face_recognition

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



def get_vggface2(path, img_size, num_classes,split_ratio=0.8, seed=1):
    print('Converting images to dataset')
    path, dirs, files = os.walk(path).__next__()
    dir_array = []
    X_train =[]
    Y_train=[]
    Y_train_label=[]
    X_test=[]
    Y_test=[]
    Y_test_label = []
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
        for f in range(len(files)):
            src = sub_path + "/" + files[f]
            img_read = cv2.imread(src)
            # cv2.imshow('o', img_read)
            face_locations = face_recognition.face_locations(img_read)
            if len(face_locations)>0:
                face_location = face_locations[0]
                b = 10
                face_height = face_location[2] - face_location[0]
                face_width = face_location[1] - face_location[3]
                img = img_read[
                                        max(face_location[0] - face_height//b, 0) : min(face_location[2]+face_height//b, img_read.shape[0]),
                                        max(face_location[3] -face_width//b,0): min(face_location[1]+face_width//b, img_read.shape[0]),
                                        :,
                                    ]

            else:
                img = img_read
            img = cv2.resize(img, (img_size, img_size))
            # cv2.imshow('c', img)
            # cv2.waitKey(1)
            img_flip_lr = cv2.flip(img, 1)
            if f < split_ratio*len(files):
                X_train.append(img)
                Y_train.append(label)
                Y_train_label.append(label_name)
                X_train.append(img_flip_lr)
                Y_train.append(label)
                Y_train_label.append(label_name)
            else:
                X_test.append(img)
                Y_test.append(label)
                Y_test_label.append(label_name)
                X_test.append(img_flip_lr)
                Y_test.append(label)
                Y_test_label.append(label_name)

    #Shuffle the data
    # random.Random(4).shuffle(data)
    test_mean = np.mean(Y_test)
    test_std = np.std(Y_test)
    train_mean = np.mean(Y_train)
    train_std = np.std(Y_train)
    cc=1
    return X_train, Y_train, np.asarray(Y_train_label), X_test, Y_test, np.asarray(Y_test_label), num_classes

def get_vggface2_mask_no_mask(path, img_size, num_classes,split_ratio=0.8, seed=1):
    print('Converting images to dataset')
    path, dirs, files = os.walk(path).__next__()
    dir_array = []
    X_train =[]
    Y_train=[]
    Y_train_label=[]
    X_test=[]
    Y_test=[]
    Y_test_label = []
    mm = min(num_classes, len(dirs))
    for i in tqdm(range(mm)):
        dir_array.append(dirs[i])
        # Read Image
        # Resize
        # Append
        # label = i
        # label_name = dirs[i]
        # label_name = ''
        sub_path = path + "/" + dirs[i]
        _, subdirs, files = os.walk(sub_path).__next__()
        num = len(files)
        mask_array =  ['surgical_blue', 'surgical_green', 'N95', 'cloth']
        for f in range(num):
            src = sub_path + "/" + files[f]
            label = 0
            label_name = 'NoMask'
            for mask_type in mask_array:
                if mask_type in files[f]:
                    label = 1
                    label_name = 'Mask'

            img = cv2.imread(src)
            img = cv2.resize(img, (img_size, img_size))
            if f < split_ratio*len(files):
                X_train.append(img)
                Y_train.append(label)
                Y_train_label.append(label_name)
            else:
                X_test.append(img)
                Y_test.append(label)
                Y_test_label.append(label_name)

    #Shuffle the data
    # random.Random(4).shuffle(data)
    test_mean = np.mean(Y_test)
    test_std = np.std(Y_test)
    train_mean = np.mean(Y_train)
    train_std = np.std(Y_train)
    cc=1
    return X_train, Y_train, np.asarray(Y_train_label), X_test, Y_test, np.asarray(Y_test_label), num_classes

def get_vggface2_raw(path, img_size, num_classes,split_ratio=0.8, seed=1):
    print('Converting images to dataset')
    path, dirs, files = os.walk(path).__next__()
    dir_array = []
    X_train =[]
    Y_train=[]
    Y_train_label=[]
    X_test=[]
    Y_test=[]
    Y_test_label = []
    # dir_array = sorted(dir_array)
    for i in tqdm(range(num_classes)):
        dir_array.append(dirs[i])
        # Read Image
        # Resize
        # Append
        label = i
        label_name = dirs[i]
        # label_name = ''
        sub_path = path + "/" + dirs[i]
        _, subdirs, files = os.walk(sub_path).__next__()
        for f in range(len(files)):
            src = sub_path + "/" + files[f]
            img_read = cv2.imread(src)
            img = cv2.resize(img_read, (img_size, img_size))

            # im = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            if f < split_ratio*len(files):
                X_train.append(img)
                Y_train.append(label)
                Y_train_label.append(label_name)
            else:
                # cv2.imshow('f', img)
                # cv2.waitKey(1)
                X_test.append(img)
                Y_test.append(label)
                Y_test_label.append(label_name)

    #Shuffle the data
    # random.Random(4).shuffle(data)
    test_mean = np.mean(Y_test)
    test_std = np.std(Y_test)
    train_mean = np.mean(Y_train)
    train_std = np.std(Y_train)
    cc=1
    return X_train, Y_train, np.asarray(Y_train_label), X_test, Y_test, np.asarray(Y_test_label), num_classes


def shuffle_dataset(data):
    random.shuffle(data)
    cc=1