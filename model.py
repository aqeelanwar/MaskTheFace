# Author: aqeelanwar
# Created: 2 May,2020, 10:48 AM
# Email: aqeel.anwar@gatech.edu
import tensorflow as tf
from network import VGGNet16


class DNN:
    def __init__(self, num_classes):
        self.g = tf.Graph()
        with self.g.as_default():
            stat_writer_path = "return_plot/"
            loss_writer_path = "loss/"
            self.stat_writer = tf.summary.FileWriter(stat_writer_path)
            self.loss_writer = tf.summary.FileWriter(loss_writer_path)
            self.num_classes = num_classes

            # Placeholders
            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.input_images = tf.placeholder(
                tf.float32, [None, None, None, 3], name="input"
            )
            self.labels = tf.placeholder(
                tf.int32, shape=[None, num_classes], name="classes"
            )

            # Preprocessing
            self.X = tf.image.resize_images(self.input_images, (224, 224))
            self.input = tf.map_fn(
                lambda frame: tf.image.per_image_standardization(frame), self.X,
            )

            self.model = VGGNet16(self.input, num_classes)

            self.output = self.model.output
            self.prediction_probs = self.model.prediction_probs
            self.predict_class = tf.argmax(self.prediction_probs, axis=1)
            self.accuracy = tf.metrics.accuracy(
                labels=self.labels, predictions=self.predict_class
            )

            self.test = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels, logits=self.output
            )
            self.loss = tf.reduce_mean(self.test)
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate, beta1=0.9, beta2=0.99
            ).minimize(self.loss, name="train_op")

            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.saver = tf.train.Saver()
            self.all_vars = tf.trainable_variables()

            self.sess.graph.finalize()

    def predict(self, input):
        labels = self.classes = tf.placeholder(tf.int32, shape=[None, self.num_classes])
        predict_class, prediction_probs = self.sess.run(
            [self.predict_class, self.prediction_probs],
            feed_dict={
                self.batch_size: input.shape[0],
                self.learning_rate: 0,
                self.input_images: input,
                self.labels: labels,
            },
        )
        return predict_class, prediction_probs

    def train(self, input, labels, lr, iter):
        _, loss, acc = self.sess.run(
            [self.train_op, self.loss, self.accuracy],
            feed_dict={
                self.batch_size: input.shape[0],
                self.learning_rate: lr,
                self.input_images: input,
                self.labels: labels,
            },
        )

        # Log to tensorboard
        self.log_to_tensorboard(
            tag="Loss", group="Main", value=loss, index=iter, type="loss"
        )
        self.log_to_tensorboard(
            tag="Acc", group="Main", value=acc, index=iter, type="loss"
        )

    def log_to_tensorboard(self, tag, group, value, index, type="loss"):
        summary = tf.Summary()
        tag = group + "/" + tag
        summary.value.add(tag=tag, simple_value=value)
        if type == "loss":
            self.loss_writer.add_summary(summary, index)
        elif type == "stat":
            self.stat_writer.add_summary(summary, index)

    def get_accuracy(self, input, labels):
        accuracy = self.sess.run(
            self.accuracy,
            feed_dict={
                self.batch_size: input.shape[0],
                self.learning_rate: 0,
                self.input_images: input,
                self.labels: labels,
            },
        )
        return accuracy

    # def evaluate(self, all_input, all_labels, batch_size):
    #     n = np.ceil(all_input.shape[0]/batch_size)
    #
    #     for i in range(n):
    #         # sample the test dataset
