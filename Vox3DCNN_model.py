import CNN3D
import tensorflow as tf
import numpy as np
import os
import shutil


class Vox3DCNN_model():
    def __init__(self):
        self.model = CNN3D.CNN3D()
        self.batch_size = 32
        self.lr = 0.00001
        self.beta = 0.5
        self.cube_len = 64
        self.features = [1, 64, 128, 256, 512, 1]
        self.shapes = np.repeat(4,len(self.features))
        self.strides = np.repeat(2, len(self.features))
        self.activation = np.append(np.repeat(CNN3D.lrelu, len(self.features)-1), tf.nn.sigmoid)
        self.normalizer = np.append(np.repeat(tf.contrib.layers.batch_norm, len(self.features) - 1), None)
        self.padding = np.append(np.repeat("SAME", len(self.features) - 1), "VALID")

        self.input = self.model.add_input([self.batch_size, self.cube_len, self.cube_len, self.cube_len, 1], 'input')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 1, 1, 1, 1], name='labels')
        with tf.variable_scope('Vox3DCNN', reuse=False):
            for i in range(1,len(self.features)):
                self.model.add_conv3d_layer(shape=[self.shapes[i], self.shapes[i], self.shapes[i], self.features[i-1], self.features[i]],
                                            strides=[1, self.strides[i], self.strides[i], self.strides[i], 1],
                                            name='conv%d'%(i+1),
                                            activation=self.activation[i],
                                            normalizer=self.normalizer[i],
                                            padding=self.padding[i])

        self.last_layer = self.model.layers[len(self.model.layers)-1].output
        self.last_layer_sigmoid = tf.nn.sigmoid(self.last_layer, 'sigmoid')
        self.last_layer_sigmoid = tf.maximum(tf.minimum(self.last_layer_sigmoid, 0.99), 0.01)
        self.summary_x_hist = tf.summary.histogram("prob_x", self.last_layer_sigmoid)

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.last_layer, labels=self.labels)
        self.loss = tf.reduce_mean(self.loss)
        self.summary_loss = tf.summary.scalar("loss", self.loss)

        self.summary = tf.summary.merge([self.summary_loss, self.summary_x_hist])

        para_d = [var for var in tf.trainable_variables() if any(x in var.name for x in ['conv', 'Vox3DCNN'])]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta).minimize(self.loss, var_list=para_d)
        self.saver = tf.train.Saver()
    def read_data(self, data_dir):
        dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        data = []
        labels = []
        label = 0
        for _dir in dirs:
            new_path = os.path.join(data_dir, _dir)
            label += 1
            files = [x for x in [f for f in os.listdir(new_path) if os.path.isfile(os.path.join(new_path, f))] if
                     '.csv' in x]
            print('loading ', _dir, '...')
            for _file in files:
                filename = os.path.join(new_path, _file)
                sample = np.zeros([self.cube_len, self.cube_len, self.cube_len])
                idxs = np.stack(
                    [x for x in np.array(np.genfromtxt(filename, delimiter=','), dtype=np.int) if x[3] > 0])[:, 0:3]
                assert np.max(idxs)<self.cube_len and np.min(idxs)>=0
                sample[idxs] = 1
                data.append(sample)
                labels.append(label)
        labels = np.asarray(labels)
        data = np.asarray(data)
        if data.shape[0]<self.batch_size:
            labels = np.repeat(labels, np.ceil(self.batch_size/labels.shape[0]), axis=0)
            data = np.repeat(data, np.ceil(self.batch_size / data.shape[0]), axis=0)
        return data, labels
    def train(self, data_dir, n_epochs = 100, checkpoint=None, is_dummy=False,):
        sess = tf.Session()
        if os.path.exists('log'):
            shutil.rmtree('log')
        os.makedirs('log')
        self.model.compile_graph('log',sess)

        if checkpoint is not None:
            self.saver.restore(sess, checkpoint)

        if is_dummy or not os.path.exists(data_dir):
            volumes = np.random.randint(0, 2, (self.batch_size, self.cube_len, self.cube_len, self.cube_len))
            labels = np.random.randint(0, 10, self.batch_size)
            print('Using Dummy Data')
        else:
            volumes, labels = self.read_data(data_dir)
            print('Using ', np.max(labels) ,' Data')
        volumes = volumes[..., np.newaxis].astype(np.float)

        for epoch in range(n_epochs):

            idx = np.random.randint(len(volumes), size=self.batch_size)
            x = volumes[idx]
            l = labels[idx].reshape((self.batch_size, 1, 1, 1, 1))

            summary, loss, _ = sess.run([self.summary, self.loss, self.optimizer], feed_dict={self.input: x, self.labels: l})
            print('Training epoch: ', epoch, ', loss:', loss)

            self.model.add_summary(summary, epoch)

            if epoch % 50 == 10:
                if not os.path.exists('model'):
                    os.makedirs('model')
                self.saver.save(sess, save_path='model/biasfree_' + str(epoch) + '.cptk')



vox = Vox3DCNN_model()
vox.train('data')
