import CNN3D
import tensorflow as tf
import numpy as np
import os
# import shutil
# from mayavi import mlab


class Vox2:
    def build_conv_graph(self, model, scope_name='Vox2', reuse=False, trainable=True):
        with tf.variable_scope(scope_name, reuse=reuse):
            for i in range(1, len(self.features)):
                model.add_conv3d_layer(
                    shape=[self.shapes[i], self.shapes[i], self.shapes[i], self.features[i - 1], self.features[i]],
                    strides=[1, self.strides[i], self.strides[i], self.strides[i], 1],
                    name='conv%d' % i,
                    activation=self.activation[i],
                    normalizer=self.normalizer[i],
                    padding=self.padding[i],
                    trainable=trainable,
                    reuse=reuse)

    def build_deconv_graph(self, conv_model, model, scope_name='Vox2', reuse=False, trainable=True):
        with tf.variable_scope(scope_name, reuse=reuse):
            for i in range(1, len(self.features)):
                model[i] = CNN3D.CNN3D()
                model[i].add_input(shape=conv_model.layers[i].output.shape.as_list(), name='layer_%d_recon' % i)
                for i2 in range(i, 0, -1):
                    model[i].add_deconv3d_layer(shape=[self.shapes[i2], self.shapes[i2], self.shapes[i2], self.features[i2 - 1], self.features[i2]],
                                                out_shape=conv_model.layers[i2-1].output.shape.as_list(),
                                                strides=[1, self.strides[i2], self.strides[i2], self.strides[i2], 1],
                                                name='conv%d' % i2,
                                                activation=None,
                                                normalizer=None,
                                                padding=self.padding[i2],
                                                trainable=trainable,
                                                reuse=True)

    def __init__(self, n_objects):

        self.model = CNN3D.CNN3D()
        self.batch_size = 32
        self.lr = 0.00001
        self.beta = 0.5
        self.cube_len = 64
        self.features = [1, 64, 128, 256, 512, 1024]
        self.shapes = np.repeat([4], len(self.features))
        self.strides = np.repeat([2], len(self.features))
        self.activation = np.append(np.repeat([CNN3D.lrelu], len(self.features) - 1), [tf.nn.sigmoid])
        self.normalizer = np.repeat([None], len(self.features))
        self.padding = np.append(np.repeat("SAME", len(self.features) - 1), "VALID")
        self.input = self.model.add_input([self.batch_size, self.cube_len, self.cube_len, self.cube_len, 1], 'input')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, n_objects], name='labels')

        self.deconv_model = {}
        self.build_conv_graph(model=self.model)
        self.build_deconv_graph(model=self.deconv_model, conv_model=self.model)

        self.last_layer, _, self.last_layer_sigmoid = self.model.add_fully_layer(features=n_objects,
                                                                                 name='full%d' % (len(self.features) - 1), reuse=False, trainable=True)
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
        self.session = tf.Session()
        self.model.compile_graph('log', self.session)

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
                assert np.max(idxs) < self.cube_len and np.min(idxs) >= 0
                for idx in idxs:
                    sample[idx[0], idx[1], idx[2]] = 1
                data.append(sample)
                labels.append(label)
        labels = np.asarray(labels)
        data = np.asarray(data)
        if data.shape[0] < self.batch_size:
            labels = np.repeat(labels, np.ceil(self.batch_size/labels.shape[0]), axis=0)
            data = np.repeat(data, np.ceil(self.batch_size / data.shape[0]), axis=0)
        labels = self.one_hot(labels, self.model.layers[len(self.model.layers)-1].output.shape.as_list()[1])
        return data, labels

    def train(self, data_dir, n_epochs=100, checkpoint=None, is_dummy=False,):

        # if os.path.exists('log'):
        #     shutil.rmtree('log')
        # os.makedirs('log')
        if not os.path.exists('model'):
            os.makedirs('model')

        if checkpoint is not None:
            self.saver.restore(self.session, checkpoint)

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
            lbl = labels[idx]

            summary, loss, _ = self.session.run([self.summary, self.loss, self.optimizer], feed_dict={self.input: x, self.labels: lbl})
            print('Training epoch: ', epoch, ', loss:', loss)

            self.model.add_summary(summary, epoch)

            if epoch % 50 == 10:
                self.saver.save(self.session, save_path='model/biasfree_' + str(epoch) + '.cptk')
        self.saver.save(self.session, save_path='model/last_model.cptk')

    def test_accuracy(self, data_dir):
        volumes, labels = self.read_data(data_dir)
        volumes = volumes[0:32]
        labels = labels[0:32]
        volumes = volumes[..., np.newaxis].astype(np.float)
        return self.session.run([self.last_layer_sigmoid, self.accuracy, self.loss],
                                feed_dict={self.input: volumes, self.labels: labels})

    def load_model(self, model):
        self.saver.restore(self.session, model)

    def save_sample(self, sample, path):
        assert sample.shape[0] == self.cube_len
        assert sample.shape[1] == self.cube_len
        assert sample.shape[2] == self.cube_len
        cntr = 0
        y = np.zeros([self.cube_len ** 3, 4])
        for x1 in np.arange(self.cube_len):
            for x2 in np.arange(self.cube_len):
                for x3 in np.arange(self.cube_len):
                    y[cntr, 0] = x1
                    y[cntr, 1] = x2
                    y[cntr, 2] = x3
                    y[cntr, 3] = sample[x1, x2, x3]
                    cntr = cntr + 1
        np.savetxt(path, y, delimiter=',')

    def reconstruct_filter(self, layer_number):
        assert layer_number in vox.deconv_model
        layer_shape = vox.deconv_model[layer_number].layers[0].output.shape.as_list()
        assert len(layer_shape) == 5
        for i1 in range(layer_shape[1]):
            for i2 in range(layer_shape[2]):
                for i3 in range(layer_shape[3]):
                    for i4 in range(layer_shape[4]):
                        z = np.zeros(layer_shape)
                        z[:, i1, i2, i3, i4] = 1
                        res = vox.session.run(vox.deconv_model[layer_number].layers[-1].output,
                                              feed_dict={vox.deconv_model[layer_number].layers[0].output: z})
                        self.save_sample(res[0], 'Reconstruct/layer%d_feature%dx%dx%dx%d.csv' % (layer_number, i1, i2, i3, i4))
                        print(i1, '/', layer_shape[1], ', ',
                              i2, '/', layer_shape[2], ', ',
                              i3, '/', layer_shape[3], ', ',
                              i4, '/', layer_shape[4], ', ', )

    @staticmethod
    def one_hot(labels, max_val=-1):
        if max_val == -1:
            max_val = np.max(labels)+1
        else:
            assert max_val >= np.max(labels)+1
        b = np.zeros((labels.shape[0], max_val))
        b[np.arange(labels.shape[0]), labels] = 1
        return b


vox = Vox2(3)
vox.load_model('model/last_model.cptk')
# vox.train('data')
vox.reconstruct_filter(5)
