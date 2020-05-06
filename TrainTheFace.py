# Author: aqeelanwar
# Created: 2 May,2020, 10:25 AM
# Email: aqeel.anwar@gatech.edu

from model import DNN
from dataset_functions import *

# data, num_classes = get_dataset(path='vgg_face_dataset/subset', img_size=224)
# X_train, Y_train, Y_train_label, X_test, Y_test, Y_test_label, num_classes = get_vggface2(path='F:\VGGface2/vggface2_train/train', img_size=224, num_classes=10)
# analyze_dataset(data, num_classes)
# X_train, Y_train, Y_train_label, X_test, Y_test, Y_test_label = split_dataset(data=data, train_ratio=0.8)
# np.save('data_split.npy', (X_train, Y_train, Y_train_label, X_test, Y_test, Y_test_label))
print('Loading data')
X_train, Y_train, Y_train_label, X_test, Y_test, Y_test_label = np.load('data_split_lr_tight.npy')
num_classes = 10
# cv2.imshow('f', np.squeeze(X_train[0]))
# cv2.waitKey(0)
# s = np.mean(Y_test)
# d = np.mean(Y_train)
print('Setting up DNN')
DNN = DNN(num_classes=num_classes)
# network_path = 'saved_network/VGG16_821/net_30.ckpt'
network_path = 'saved_network/net_1.ckpt'
DNN.load_network(network_path)
batch_size = 16

iteration = 0
lr = 1e-5
num_epochs = 100

for epoch in range(num_epochs):
    if epoch>0:
        lr /=epoch
    # Shuffle the dataset
    num_ones = np.sum(Y_train)
    # cv2.imshow('c', np.squeeze(X_train[1]))

    random.Random(4).shuffle(X_train)
    random.Random(4).shuffle(Y_train)
    # cv2.imshow('shuffle', np.squeeze(X_train[1]))
    num_ones = np.sum(Y_train)
    # cv2.waitKey(0)
    iteration += 1
    n = int(np.ceil(len(X_train) / batch_size))
    m = int(np.ceil(len(X_test) / batch_size))
    for i in range(n):
        input = np.asarray(X_train[i * batch_size: (i + 1) * batch_size])
        labels = Y_train[i * batch_size: (i + 1) * batch_size]
        labels = np.asarray([labels]).T
        loss, acc = DNN.train(input, labels, lr, iteration, keep_prob=0.8)
        print('Epoch: {:3} Iter: {:3}/{:3} Loss: {:2.6f} Acc: {:2.3f}'.format(epoch, i, n-1, loss, acc))
    # At the end of each epoch, show test accuracy
    test_acc = 0
    random.Random(4).shuffle(X_test)
    random.Random(4).shuffle(Y_test)
    for j in range(m):
        # TODO when num remaining elements < batchsize
        input = np.asarray(X_test[j * batch_size : (j + 1) * batch_size])
        labels = Y_test[j * batch_size : (j + 1) * batch_size]
        labels = np.asarray([labels]).T
        acc = DNN.get_accuracy(input, labels)
        test_acc = test_acc + acc * input.shape[0]

    test_acc /= len(X_test)
    DNN.log_to_tensorboard(tag='Test Acc', group='Test', value=test_acc, index=epoch)
    print('--------------------------------------------------------')
    print("Epoch: {:3} Test Acc: {:2.3f}".format(epoch, test_acc))
    # Save the current network
    DNN.save_network(epoch)
    print('--------------------------------------------------------')

