# Author: aqeelanwar
# Created: 2 May,2020, 10:25 AM
# Email: aqeel.anwar@gatech.edu

from model import DNN
from dataset_functions import *

data, num_classes = get_dataset(path='vgg_face_dataset/subset', img_size=224)
X_train, Y_train, Y_train_label, X_test, Y_test, Y_test_label = split_dataset(data=data, train_ratio=0.8)

DNN = DNN(num_classes=num_classes)
batch_size = 32

iteration = 0
lr = 1e-6
num_epochs = 100

for epoch in range(num_epochs):
    # Shuffle the dataset
    random.Random(4).shuffle(X_train)
    random.Random(4).shuffle(Y_train)
    iteration += 1
    n = int(np.ceil(Y_train.shape[0] / batch_size))
    m = int(np.ceil(Y_test.shape[0] / batch_size))
    for i in range(n):
        input = X_train[i * batch_size: (i + 1) * batch_size]
        labels = Y_train[i * batch_size: (i + 1) * batch_size]
        loss, acc = DNN.train(input, labels, lr, iteration, keep_prob=0.8)
        print('Epoch: {:3} Iter: {:3}/{:3} Loss: {:2.6f} Acc: {:2.3f}'.format(epoch, i, n-1, loss, acc))
    # At the end of each epoch, show test accuracy
    test_acc = 0
    for j in range(m):
        input = X_test[j * batch_size : (j + 1) * batch_size]
        labels = Y_test[j * batch_size : (j + 1) * batch_size]
        acc = DNN.get_accuracy(input, labels)
        test_acc = test_acc + acc * input.shape[0]

    test_acc /= X_test.shape[0]
    DNN.log_to_tensorboard(tag='Test Acc', group='Test', value=test_acc, index=epoch)
    print('-------------------------------------------------')
    print("Epoch: {:3} Test Acc: {:2.3f}".format(epoch, test_acc))
    print('-------------------------------------------------')

    # Save the current network
    DNN.save_network(epoch)
