# Author: Aqeel Anwar(ICSRL)
# Created: 5/4/2020, 9:55 AM
# Email: aqeel.anwar@gatech.edu

from model import DNN
from dataset_functions import *

# data, num_classes = get_dataset(path='vgg_face_dataset/subset', img_size=224)
X_train, Y_train, Y_train_label, X_test, Y_test, Y_test_label, num_classes = get_vggface2(path='F:\VGGface2/vggface2_train/subset_masked', img_size=224, num_classes=10)
# X_train, Y_train, Y_train_label, X_test, Y_test, Y_test_label = np.load('data_split_lr_tight.npy')
num_classes = 10
print('Setting up DNN')
DNN = DNN(num_classes=num_classes)
# network_path = 'saved_network/net_2.ckpt'
network_path = 'saved_network\VGG16_test687_lrn/net_7.ckpt'
DNN.load_network(network_path)
batch_size = 32
X = X_test
Y = Y_test
m = int(np.ceil(len(X) / batch_size))
test_acc = 0
print('Calculating accuracy')
# random.Random(10).shuffle(X_test)
# random.Random(10).shuffle(Y_test)
for j in tqdm(range(m)):
    input = np.asarray(X[j * batch_size: (j + 1) * batch_size])
    labels = Y[j * batch_size: (j + 1) * batch_size]
    labels = np.asarray([labels]).T
    acc = DNN.get_accuracy(input, labels)
    test_acc = test_acc + acc * input.shape[0]

test_acc /= len(X)
print('--------------------------------------------------------')
print("Test Acc: {:2.3f}".format(test_acc))
print('--------------------------------------------------------')