import CNN3D
import tensorflow as tf
import numpy as np
import os
import shutil
from mayavi import mlab


class Vox3DCNN_model():
    def __init__(self, n_objects):
        self.model = CNN3D.CNN3D()
        self.batch_size = 32
        self.lr = 0.00001
        self.beta = 0.5
        self.cube_len = 64
        self.features = [1, 64, 128, 256, 512, 1024]
        self.shapes = np.repeat(4,len(self.features))
        self.strides = np.repeat(2, len(self.features))
        self.activation = np.append(np.repeat(CNN3D.lrelu, len(self.features)-1), tf.nn.sigmoid)
        self.normalizer = np.append(np.repeat(tf.contrib.layers.batch_norm, len(self.features) - 1), None)
        self.padding = np.append(np.repeat("SAME", len(self.features) - 1), "VALID")

        self.input = self.model.add_input([self.batch_size, self.cube_len, self.cube_len, self.cube_len, 1], 'input')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, n_objects], name='labels')
        with tf.variable_scope('Vox3DCNN', reuse=False):
            for i in range(1,len(self.features)):
                self.model.add_conv3d_layer(shape=[self.shapes[i], self.shapes[i], self.shapes[i], self.features[i-1], self.features[i]],
                                            strides=[1, self.strides[i], self.strides[i], self.strides[i], 1],
                                            name='conv%d'%(i+1),
                                            activation=self.activation[i],
                                            normalizer=self.normalizer[i],
                                            padding=self.padding[i])

        self.last_layer, _, self.last_layer_sigmoid = self.model.add_fully_layer(n_objects, 'full%d'%(len(self.features)-1))
        self.correct_predictions = tf.equal(tf.argmax(self.last_layer_sigmoid, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"))
        self.summary_accuracy = tf.summary.scalar("prob_x", self.accuracy)

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.last_layer, labels=self.labels)
        self.loss = tf.reduce_mean(self.loss)
        self.summary_loss = tf.summary.scalar("loss", self.loss)

        self.summary = tf.summary.merge([self.summary_loss, self.summary_accuracy])

        para_d = [var for var in tf.trainable_variables() if any(x in var.name for x in ['conv', 'Vox3DCNN'])]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta).minimize(self.loss, var_list=para_d)
        self.saver = tf.train.Saver()
    def read_data(self, data_dir):
        dirs = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        data = []
        labels = []
        label = -1
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
                for idx in idxs:
                    sample[idx[0],idx[1],idx[2]] = 1
                data.append(sample)
                labels.append(label)
        labels = np.asarray(labels)
        data = np.asarray(data)
        if data.shape[0]<self.batch_size:
            labels = np.repeat(labels, np.ceil(self.batch_size/labels.shape[0]), axis=0)
            data = np.repeat(data, np.ceil(self.batch_size / data.shape[0]), axis=0)
        labels = self.onehot(labels, self.model.layers[len(self.model.layers)-1].output.shape.as_list()[1])
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
        volumes = volumes[..., np.newaxis].astype(np.float)

        for epoch in range(n_epochs):

            idx = np.random.randint(len(volumes), size=self.batch_size)
            x = volumes[idx]
            l = labels[idx]

            summary, loss, _ = sess.run([self.summary, self.loss, self.optimizer], feed_dict={self.input: x, self.labels: l})
            print('Training epoch: ', epoch, ', loss:', loss)

            self.model.add_summary(summary, epoch)

            if epoch % 50 == 10:
                if not os.path.exists('model'):
                    os.makedirs('model')
                self.saver.save(sess, save_path='model/biasfree_' + str(epoch) + '.cptk')
    def render3D(self, data):
        src = mlab.pipeline.scalar_field(data)
        outer = mlab.pipeline.iso_surface(src)
        mlab.show()
    def onehot(self, labels, max_val=-1):
        if(max_val==-1):
            max_val = np.max(labels)+1
        else:
            assert max_val>=np.max(labels)+1
        b = np.zeros((labels.shape[0], max_val))
        b[np.arange(labels.shape[0]), labels] = 1
        return b



vox = Vox3DCNN_model(3)
vox.train('data')