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
        self.cube_len = 32
        self.features = [1, 64, 128, 256, 512, 1024]
        self.shapes = [0, 6, 5, 4, 3, 2]
        self.strides = np.repeat([2], len(self.features))
        self.activation = np.append(np.repeat([CNN3D.lrelu], len(self.features) - 1), [tf.nn.sigmoid])
        self.normalizer = np.repeat([None], len(self.features))
        # np.append(np.repeat(tf.contrib.layers.batch_norm, len(self.features) - 1), [None])
        # self.normalizer = np.repeat([None], len(self.features))
        self.padding = np.append(np.repeat("SAME", len(self.features) - 1), "VALID")
        self.input = self.model.add_input([self.batch_size, self.cube_len, self.cube_len, self.cube_len, 1], 'input')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, n_objects], name='labels')

        self.deconv_model = {}
        self.build_conv_graph(model=self.model)
        self.build_deconv_graph(model=self.deconv_model, conv_model=self.model)

        # self.model.add_fully_layer(features=1024,
        #                            name='full%d' % len(self.model.layers), reuse=False, trainable=True)

        self.last_layer, _, self.last_layer_sigmoid = self.model.add_fully_layer(features=n_objects,
                                                                                 name='full%d' % len(self.model.layers), reuse=False, trainable=True)
        self.correct_predictions = tf.equal(tf.argmax(self.last_layer_sigmoid, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"))
        self.summary_accuracy_training = tf.summary.scalar("Accuracy", self.accuracy)
        self.summary_accuracy_testing = tf.summary.scalar("Accuracy", self.accuracy)

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.last_layer, labels=self.labels)
        self.loss = tf.reduce_mean(self.loss)
        self.summary_loss_training = tf.summary.scalar("Loss", self.loss)
        self.summary_loss_testing = tf.summary.scalar("Loss", self.loss)

        self.summary_training = tf.summary.merge([self.summary_loss_training, self.summary_accuracy_training])
        self.summary_testing = tf.summary.merge([self.summary_loss_testing, self.summary_accuracy_testing])

        para_d = [var for var in tf.trainable_variables() if any(x in var.name for x in ['conv', 'Vox3DCNN'])]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta).minimize(self.loss, var_list=para_d)
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.model.compile_graph('log', self.session)

    def read_model_net(self, data_file, nsamples=-1):
        data = []
        labels = []
        x_ = np.load(data_file)['features']
        y_ = np.load(data_file)['targets']

        if nsamples == -1 and len(x_) < 100000:
            for i in range(len(x_)):
                data.append(np.reshape(x_[i], [32, 32, 32]))
                labels.append([y_[i]])
        else:
            if nsamples == -1:
                nsamples = 100000
            assert nsamples < len(x_)
            u_, i_ = np.unique(y_, return_index=True)
            samples_per_class = np.floor(nsamples/len(i_)).astype(np.int)
            for i in range(len(i_)):
                for k in range(samples_per_class):
                    if i_[i] + k >= len(x_):
                        break
                    if i+1 < len(i_):
                        if i_[i] + k >= i_[i+1]:
                            break
                    data.append(np.reshape(x_[i_[i] + k], [32, 32, 32]))
                    labels.append([u_[i]])
        labels = np.asarray(labels)
        data = np.asarray(data)
        if data.shape[0] < self.batch_size:
            labels = np.repeat(labels, np.ceil(self.batch_size/labels.shape[0]), axis=0)
            data = np.repeat(data, np.ceil(self.batch_size / data.shape[0]), axis=0)

        labels = self.one_hot(labels, self.model.layers[len(self.model.layers)-1].output.shape.as_list()[1])
        data = data[..., np.newaxis].astype(np.float)
        return data, labels

    def train(self, data_dir, nsamples=-1, n_epochs=100, checkpoint=None, is_dummy=False,):

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
            volumes, labels = self.read_model_net(data_dir, nsamples)

        idx = np.random.permutation(len(volumes))
        x = volumes[idx]
        lbl = labels[idx]
        iters = np.floor(len(volumes)*0.2/self.batch_size).astype(np.int)
        for epoch in range(n_epochs):
            for iteration in range(iters):
                self.session.run(self.optimizer, feed_dict={self.input: x[iteration*self.batch_size:(iteration+1)*self.batch_size],
                                                            self.labels: lbl[iteration*self.batch_size:(iteration+1)*self.batch_size]})
            summary, loss, acc = self.session.run([self.summary, self.loss, self.accuracy],
                                                  feed_dict={self.input: x[iteration*self.batch_size:(iteration+1)*self.batch_size],
                                                             self.labels: lbl[iteration*self.batch_size:(iteration+1)*self.batch_size]})
            print('Training epoch:', epoch, ', loss:', loss, ', accuracy:', acc)
            self.model.add_summary(summary, epoch)

            if epoch % 50 == 10:
                self.saver.save(self.session, save_path='model/biasfree_' + str(epoch) + '.cptk')
        self.saver.save(self.session, save_path='model/last_model.cptk')

    @staticmethod
    def shuffle_data(x, lbl):
        idx = np.random.permutation(len(x))
        x = x[idx]
        lbl = lbl[idx]
        return x, lbl

    def intensive_train(self, training_file, testing_file):

        if not os.path.exists('model'):
            os.makedirs('model')

        n_epochs = 10000
        training_x, training_l = self.read_model_net(training_file)
        testing_x, testing_l = self.read_model_net(testing_file)
        iters = np.floor(len(training_x)*0.2/self.batch_size).astype(np.int)

        for epoch in range(n_epochs):
            training_x, training_l = self.shuffle_data(training_x, training_l)
            testing_x, testing_l = self.shuffle_data(testing_x, testing_l)
            for iteration in range(iters):
                self.session.run(self.optimizer, feed_dict={self.input: training_x[iteration*self.batch_size:(iteration+1)*self.batch_size],
                                                            self.labels: training_l[iteration*self.batch_size:(iteration+1)*self.batch_size]})
            summary_training, loss, acc = self.session.run([self.summary_training, self.loss, self.accuracy],
                                                           feed_dict={self.input: training_x[:self.batch_size],
                                                                      self.labels: training_l[:self.batch_size]})
            summary_testing, loss2, acc2 = self.session.run([self.summary_testing, self.loss, self.accuracy],
                                                            feed_dict={self.input: testing_x[:self.batch_size],
                                                                       self.labels: testing_l[:self.batch_size]})
            print('Training epoch:', epoch, ', loss:', loss, ', accuracy:', acc)
            print('Testing epoch:', epoch, ', loss:', loss2, ', accuracy:', acc2)
            self.model.add_summary(summary_training, epoch)
            self.model.add_summary(summary_testing, epoch)

            if epoch % 50 == 10:
                self.saver.save(self.session, save_path='model/biasfree_' + str(epoch) + '.cptk')

        self.saver.save(self.session, save_path='model/last_model.cptk')

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
        for k in range(labels.shape[0]):
            b[k, labels[k]] = 1
        return b

    def test_data(self, data_dir, nsamples=-1):
        assert os.path.exists(data_dir)
        volumes, labels = self.read_model_net(data_dir, nsamples)
        iters = np.floor(len(volumes)/self.batch_size).astype(np.int)
        acc = 0.0
        for iteration in range(iters):
            nacc = self.session.run(self.accuracy,
                                    feed_dict={self.input: volumes[iteration*self.batch_size:(iteration+1)*self.batch_size],
                                               self.labels: labels[iteration*self.batch_size:(iteration+1)*self.batch_size]})
            # print('batch:', iteration, ', accuracy:', nacc)
            acc += nacc / iters
        print('Mean Accuracy:', acc)
        return acc


vox = Vox2(40)
# vox.load_model('data/ModelNet40/pretrained/biasfree_10.cptk')
# vox.test_data('data/ModelNet40/modelnet40_rot_train.npz', nsamples=1000)
# vox.test_data('data/ModelNet40/modelnet40_rot_test.npz', nsamples=-1) # 29616
# vox.train('data/ModelNet40/modelnet40_rot_train.npz', n_epochs=600, nsamples=100000)
# vox.reconstruct_filter(5)

vox.intensive_train(training_file='data/ModelNet40/modelnet40_rot_train.npz',
                    testing_file='data/ModelNet40/modelnet40_rot_test.npz')
