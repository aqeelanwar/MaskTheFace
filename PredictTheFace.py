# Author: Aqeel Anwar(ICSRL)
# Created: 5/4/2020, 9:55 AM
# Email: aqeel.anwar@gatech.edu

from model import DNN
from dataset_functions import *

# data, num_classes = get_dataset(path='vgg_face_dataset/subset', img_size=224)
data, num_classes = get_vggface2(path='F:\VGGface2/vggface2_train/train', img_size=224, num_classes=20)
X_train, Y_train, Y_train_label, X_test, Y_test, Y_test_label = split_dataset(data=data, train_ratio=0.8)

print('Setting up DNN')
DNN = DNN(num_classes=num_classes)
network_path = 'saved_network/VGG16_821/net_30.ckpt'
# network_path = 'saved_network/net_0.ckpt'
DNN.load_network(network_path)
batch_size = 16
m = int(np.ceil(Y_test.shape[0] / batch_size))
test_acc = 0
print('Calculating accuracy')
for j in tqdm(range(m)):
    input = X_test[j * batch_size : (j + 1) * batch_size]
    labels = Y_test[j * batch_size : (j + 1) * batch_size]
    acc = DNN.get_accuracy(input, labels)
    predict_class, prediction_probs = DNN.predict(input)
    print('Batch Acc: ', acc)
    test_acc = test_acc + acc * input.shape[0]

test_acc /= X_test.shape[0]
print('--------------------------------------------------------')
print("Test Acc: {:2.3f}".format(test_acc))
print('--------------------------------------------------------')