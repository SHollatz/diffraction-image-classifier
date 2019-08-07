from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import tensorflow as tf
import pandas as pd
import resource
from datetime import datetime
from sklearn.metrics import confusion_matrix
import keras as k


class Train:

    def __init__(self):
        self.__x_ = None
        self.__y_ = None
        self.__logits = None
        self.__loss = None
        self.__loss_val = None
        self.__loss_test = None
        self.__step_variable = None
        self.__saver = None
        self.__session = None
        self.__writer_train = None
        self.__writer_val = None
        self.__writer_test = None
        self.__is_training = None
        self.__merged_summary_train = None
        self.__merged_summary_val = None
        self.__merged_summary_test = None
        self.__optimizer = None
        self.__accuracy = None
        self.__val_accuracy = None
        self.__test_accuracy = None

    def build_graph(self):
        self.__x_ = tf.placeholder("float", shape=[None, 224, 224, 1], name="X")
        self.__y_ = tf.placeholder("int32", shape=[None, 5], name="Y")
        self.__is_training = tf.placeholder(tf.bool)

        # VGG16 according to book Hands-On CNNs with TF
        with tf.name_scope("model"):
            conv1_1 = k.layers.Conv2D(filters=64, kernel_size=[3, 3], input_shape=(None, 224, 224, 1),
                                      padding="same", activation=k.layers.activations.relu,
                                      data_format="channels_last", name="conv1_1")(self.__x_)
            conv1_2 = k.layers.Conv2D(filters=64, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv1_2")(conv1_1)
            pool1 = k.layers.MaxPooling2D(pool_size=[2, 2], strides=2, name="pool1")(conv1_2)
            conv2_1 = k.layers.Conv2D(filters=128, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv2_1")(pool1)
            conv2_2 = k.layers.Conv2D(filters=128, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv2_2")(conv2_1)
            pool2 = k.layers.MaxPooling2D(pool_size=[2, 2], strides=2, name="pool2")(conv2_2)
            conv3_1 = k.layers.Conv2D(filters=256, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv3_1")(pool2)
            conv3_2 = k.layers.Conv2D(filters=256, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv3_2")(conv3_1)
            conv3_3 = k.layers.Conv2D(filters=256, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv3_3")(conv3_2)
            pool3 = k.layers.MaxPooling2D(pool_size=[2, 2], strides=2, name="pool3")(conv3_3)
            conv4_1 = k.layers.Conv2D(filters=512, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv4_1")(pool3)
            conv4_2 = k.layers.Conv2D(filters=512, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv4_2")(conv4_1)
            conv4_3 = k.layers.Conv2D(filters=512, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv4_3")(conv4_2)
            pool4 = k.layers.MaxPooling2D(pool_size=[2, 2], strides=2, name="pool4")(conv4_3)
            conv5_1 = k.layers.Conv2D(filters=512, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv5_1")(pool4)
            conv5_2 = k.layers.Conv2D(filters=512, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv5_2")(conv5_1)
            conv5_3 = k.layers.Conv2D(filters=512, kernel_size=[3, 3],
                                       padding="same", activation=tf.nn.relu, name="conv5_3")(conv5_2)
            pool5 = k.layers.MaxPooling2D(pool_size=[2, 2], strides=2, name="pool5")(conv5_3)
            # Flatten input data
            pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
            # FC Layers (can be removed)
            fc6 = k.layers.Dense(units=4096, activation=tf.nn.relu, name="fc6")(pool5_flat)
            fc7 = k.layers.Dense(units=4096, activation=tf.nn.relu, name="fc7")(fc6)
            # output num_categories
            self.__logits = k.layers.Dense(units=5, activation=None, name="logits")(fc7)
            # predictions = tf.nn.softmax(fc8, name='predictions')

        # log graph
        tf.logging.info("Graph Summary:")
        tf.logging.info("input: %s" % str(self.__x_.shape.as_list()))
        tf.logging.info("conv1_1 %s" % str(conv1_1.shape.as_list()))
        tf.logging.info("conv1_2 %s" % str(conv1_2.shape.as_list()))
        tf.logging.info("pool1 %s" % str(pool1.shape.as_list()))
        tf.logging.info("conv2_1 %s" % str(conv2_1.shape.as_list()))
        tf.logging.info("conv2_2 %s" % str(conv2_2.shape.as_list()))
        tf.logging.info("pool2 %s" % str(pool2.shape.as_list()))
        tf.logging.info("conv3_1 %s" % str(conv3_1.shape.as_list()))
        tf.logging.info("conv3_2 %s" % str(conv3_2.shape.as_list()))
        tf.logging.info("conv3_3 %s" % str(conv3_3.shape.as_list()))
        tf.logging.info("pool3 %s" % str(conv3_3.shape.as_list()))
        tf.logging.info("conv4_1 %s" % str(conv4_1.shape.as_list()))
        tf.logging.info("conv4_2 %s" % str(conv4_2.shape.as_list()))
        tf.logging.info("conv4_3 %s" % str(conv4_3.shape.as_list()))
        tf.logging.info("pool4 %s" % str(pool4.shape.as_list()))
        tf.logging.info("conv5_1 %s" % str(conv5_1.shape.as_list()))
        tf.logging.info("conv5_2 %s" % str(conv5_2.shape.as_list()))
        tf.logging.info("conv5_3 %s" % str(conv5_3.shape.as_list()))
        tf.logging.info("pool5 %s" % str(pool5.shape.as_list()))
        tf.logging.info("pool5_flat %s" % str(pool5_flat.shape.as_list()))
        tf.logging.info("fc6 %s" % str(fc6.shape.as_list()))
        tf.logging.info("fc7 %s" % str(fc7.shape.as_list()))
        tf.logging.info("logits %s" % str(self.__logits.shape.as_list()))

        # Compute loss
        with tf.name_scope("loss_func"):
            self.__loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=self.__logits, labels=self.__y_))
            self.__loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=self.__logits, labels=self.__y_))
            self.__loss_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=self.__logits, labels=self.__y_))

        # Create optimizer
        with tf.name_scope("optimizer"):
            # Define an untrainable variable to hold the global step (every batch computation)
            self.__step_variable = tf.Variable(0., name="global_step", trainable=False)

            starter_learning_rate = 0.001

            # decay every 10000 steps with a base of 0.96 function#
            # although the Adam optimizer automatically adjusts and decays the lr, the scheduling may to improve results
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.__step_variable, 250, 0.9,
                                                       staircase=True)
            self.__optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.__loss,
                                                                              global_step=self.__step_variable)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.__logits, 1), tf.argmax(self.__y_, 1))
            with tf.name_scope('accuracy'):
                self.__accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self.__val_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self.__test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Visualize in tensorboard
        summaries_train = [tf.summary.scalar("learning_rate", learning_rate),
                           tf.summary.scalar("global_step", self.__step_variable),
                           tf.summary.scalar("loss_train", self.__loss),
                           tf.summary.scalar('accuracy', self.__accuracy)]
        summaries_val = [tf.summary.scalar('val_accuracy', self.__val_accuracy),
                         tf.summary.scalar("loss_val", self.__loss_val)]
        summaries_test = [tf.summary.scalar('test_accuracy', self.__test_accuracy),
                          tf.summary.scalar("loss_test", self.__loss_test)]

        self.__merged_summary_train = tf.summary.merge(summaries_train)
        self.__merged_summary_val = tf.summary.merge(summaries_val)
        self.__merged_summary_test = tf.summary.merge(summaries_test)

        # initializer for variables
        init = tf.global_variables_initializer()

        # Saver for checkpoints
        self.__saver = tf.train.Saver(max_to_keep=None)

        # Avoid allocating the whole memory
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        self.__session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        k.backend.set_session(self.__session)

        # Configure summary to output at given directory
        logdir = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.__writer_train = tf.summary.FileWriter("./logs/logs_" + logdir + "/debug_train_", self.__session.graph)
        self.__writer_val = tf.summary.FileWriter("./logs/logs_" + logdir + "/debug_val", self.__session.graph)
        self.__writer_test = tf.summary.FileWriter("./logs/logs_" + logdir + "/debug_test", self.__session.graph)

        # initialize variables
        self.__session.run(init)

    def preprocess_image(self, img):
        img_arr = tf.image.decode_png(img, channels=1)
        image = img_arr / 255  # normalize to [0,1]

        return image

    def load_and_preprocess_image(self, filename):
        image = tf.read_file(tf.cast(filename, dtype=tf.string))

        return self.preprocess_image(image)

    def create_data_iterator(self, num_imgs, img_data_array, input_path, batch_size):
        img_paths = []
        for i in range(num_imgs):
            img_paths.append(input_path + img_data_array[i][0])
            
        path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
        image_ds = path_ds.map(self.load_and_preprocess_image)
        
        onehot_labels = []
        for i in range(num_imgs):
            onehot_labels.append(img_data_array[i][1])

        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(onehot_labels, tf.int64))
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

        # Enable Logging
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # shuffle and batch
        final_ds = image_label_ds.shuffle(buffer_size=num_imgs)
        final_ds = final_ds.batch(batch_size)
        final_ds = final_ds.prefetch(buffer_size=num_imgs)

        # Create an iterator
        iterator = final_ds.make_initializable_iterator()
        return iterator

    def train(self):
        num_epochs = 1000
        batch_size_train = 89
        batch_size_val = 20
        batch_size_test = 85
        num_images = 12503
        num_train_images = 8010
        num_val_images = 2000  # left 12 out to make divisible by batch_size_val
        num_test_images = 2380  # left one out to make divisible for batch_size
        num_batches_train = int(num_train_images / batch_size_train)  # 90
        num_batches_test = int(num_test_images / batch_size_test)  # 28
        path = "/data/holton/deep_learning/try1/resnet1/"
        data = pd.read_csv(path + "categories_multi.txt", delim_whitespace=True, header=None)

        all_img_data = []
        for i in range(num_images):
            single_image_data = []
            single_image_labels = [
                data.at[i, 1], 
                data.at[i, 2], 
                data.at[i, 3], 
                data.at[i, 4], 
                data.at[i, 5]
            ]
            single_image_data.append(data.at[i, 0])
            single_image_data.append(single_image_labels)
            all_img_data.append(single_image_data)

        train_img_data = []
        val_img_data = []
        test_img_data = []
        for img_data in all_img_data:
            if "training" in img_data[0]:
                train_img_data.append(img_data)
            elif "validation" in img_data[0]:
                val_img_data.append(img_data)
            else:
                test_img_data.append(img_data)

        iter_train = self.create_data_iterator(num_train_images, train_img_data, path, batch_size_train)
        iter_train_op = iter_train.get_next()

        iter_val = self.create_data_iterator(num_val_images, val_img_data, path, batch_size_val)
        iter_val_op = iter_val.get_next()

        iter_test = self.create_data_iterator(num_test_images, test_img_data, path, batch_size_test)
        iter_test_op = iter_test.get_next()

        self.build_graph()

        # Train Loop
        for epoch in range(1, num_epochs+1):
            logging.info("")
            logging.info("Before epoch %s: memory usage: %s (gb)",
                         epoch, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
            self.__session.run(iter_train.initializer)
            self.__session.run(iter_val.initializer)

            for batch in range(1, num_batches_train+1):  # range(90)
                # Perform training step
                try:
                    batch_train = self.__session.run([iter_train_op])
                    batch_x_train, batch_y_train = batch_train[0]

                    # Execute train op
                    _, loss_train, accuracy, summary_train, step = self.__session.run([self.__optimizer, self.__loss,
                                                    self.__accuracy, self.__merged_summary_train, self.__step_variable],
                                                    feed_dict={self.__x_: batch_x_train, self.__y_: batch_y_train,
                                                               self.__is_training: True})

                except tf.errors.OutOfRangeError:
                    logging.info("The iterator is out of range.")
                    break

            batch_val = self.__session.run([iter_val_op])
            batch_x_val, batch_y_val = batch_val[0]
            loss_val, val_accuracy, summary_val = self.__session.run([self.__loss_val, self.__val_accuracy,
                                                                     self.__merged_summary_val],
                                                                     feed_dict={self.__x_: batch_x_val,
                                                                                self.__y_: batch_y_val,
                                                                                self.__is_training: False})

            # display images in tensorboard
            images = np.reshape(batch_x_train[:], (-1, 224, 224, 1))
            image_summary_op = tf.summary.image("train_batch", images, max_outputs=89)
            image_summary = self.__session.run(image_summary_op, feed_dict={self.__x_: batch_x_train,
                                                                            self.__y_: batch_y_train,
                                                                            self.__is_training: False})
            # Write to tensorboard summary
            self.__writer_train.add_summary(image_summary, step)
            self.__writer_train.add_summary(summary_train, step)
            self.__writer_val.add_summary(summary_val, step)

            logging.info("After training epoch %s: memory usage: %s (gb)",
                         epoch, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024)
            logging.info("Gobal Step: %s", step)
            logging.info("Loss Train: %s, Loss Val: %s" % (loss_train, loss_val))

            logging.info("Acc Train: %s, Acc Val: %s" % (accuracy, val_accuracy))

        self.__session.run(iter_test.initializer)
        test_logits = []
        test_predictions = []

        for j in range(num_batches_test):
            batch_test = self.__session.run([iter_test_op])
            batch_x_test, batch_y_test = batch_test[0]

            loss_test, test_accuracy, summary_test, logits = self.__session.run(
                [self.__loss_test, self.__test_accuracy, self.__merged_summary_test, self.__logits],
                feed_dict={self.__x_: batch_x_test,
                           self.__y_: batch_y_test,
                           self.__is_training: False})

            for pred in logits:
                test_logits.append(pred)
                test_predictions.append(np.argmax(pred))
                self.__writer_test.add_summary(summary_test, step)

            logging.info("Loss Test: {0} Acc Test: {1}".format(loss_test, test_accuracy))
            logging.info("")

        onehot_labels_test = []
        for i in range(num_test_images):
            onehot_labels_test.append(test_img_data[i][1])    
        class_labels = []
        for label in onehot_labels_test:
            class_labels.append(np.argmax(label))

        # calculate confusion matrix
        titel = 'VGG16_try1_classifier'
        cm_plot_labels = ['blank', 'good', 'noxtal', 'strong', 'weak']
        tick_marks = np.arange(5)

        cm = confusion_matrix(class_labels, test_predictions)
        logging.info(cm)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format=("%(levelname)s %(asctime)s "
                                "%(filename)s:%(lineno)d "    
                                "%(funcName)s(): "
                                "%(message)s"),
                        datefmt="%m%d %H:%M:%S")
    cnn = Train()
    cnn.train()
