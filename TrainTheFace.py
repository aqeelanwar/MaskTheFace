# Author: aqeelanwar
# Created: 2 May,2020, 10:25 AM
# Email: aqeel.anwar@gatech.edu

from model import DNN

num_classes = 10
DNN = DNN(num_classes=num_classes)
batch_size = 32

iteration = 0
lr = 1e-4
num_epochs = 100


for epoch in range(num_epochs):
    iter += 1
    n = input_array_train.shape[0] / batch_size
    m = input_array_test.shape[0] / batch_size
    for i in range(n):
        # sample batch
        input = input_array_train[i * batch_size : (i + 1) * batch_size]
        labels = labels_array_train[i * batch_size : (i + 1) * batch_size]
        DNN.train(input, labels, lr, iteration)

    # At the end of each epoch, show test accuracy
    test_acc = 0
    for j in range(m):
        input = input_array_test[i * batch_size : (i + 1) * batch_size]
        labels = labels_array_test[i * batch_size : (i + 1) * batch_size]
        acc = DNN.get_accuracy(input, labels)
        test_acc += acc * input.shape[0]

    test_acc /= input_array_test.shape[0]
    print("Epoch: ", epoch, " Test Acc: ", test_acc)
