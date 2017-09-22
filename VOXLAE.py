import CNN3D
import tensorflow as tf
import numpy as np
import os
from shutil import copyfile

class VOXLAE:
    def build_VOXLAE_graph(self, model, reuse=False, trainable=True):
        for i in range(1, len(self.features)):
            print('Adding ', i, ' layer')
            model.add_up_down_ae(shape=[self.shapes[i], self.shapes[i], self.shapes[i], self.features[i - 1], self.features[i]],
                                 strides=[1, self.strides[i], self.strides[i], self.strides[i], 1],
                                 name='Layer%d' % i,
                                 up_activation=self.activation[i],
                                 down_activation=self.activation[i],
                                 up_normalizer=self.normalizer[i],
                                 down_normalizer=self.normalizer[i],
                                 up_padding=self.padding[i],
                                 down_padding=self.padding[i],
                                 up_trainable=trainable,
                                 down_trainable=trainable,
                                 reuse=reuse,
                                 lr=self.lr,
                                 beta=self.beta)

    def __init__(self, n_objects):
        self.model = CNN3D.CNN3D()
        self.batch_size = 32
        self.lr = 0.00001
        self.beta = 0.5
        self.cube_len = 32
        self.features = [1, 64, 128, 256, 512, 1024]
        self.shapes = [0, 6, 5, 4, 3, 2]
        self.strides = np.repeat([2], len(self.features))
        # self.activation = np.repeat([CNN3D.lrelu], len(self.features))
        self.activation = np.repeat([tf.nn.sigmoid], len(self.features))
        # self.normalizer = np.repeat([None], len(self.features))
        # self.normalizer = np.append(np.repeat(tf.contrib.layers.batch_norm, len(self.features) - 1), [None])
        self.normalizer = np.repeat(tf.contrib.layers.batch_norm, len(self.features))
        self.padding = np.append(np.repeat("SAME", len(self.features) - 1), "VALID")

        with tf.variable_scope('VOXLAE'):
            self.input = self.model.add_input([self.batch_size, self.cube_len, self.cube_len, self.cube_len, 1], 'input')
            self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, n_objects], name='labels')
            self.build_VOXLAE_graph(model=self.model)
            temp = tf.trainable_variables()
            self.last_layer, _, self.last_layer_sigmoid = self.model.add_fully_layer(features=n_objects,
                                                                                     name='full%d' % len(self.model.layers), reuse=False, trainable=True)
            self.correct_predictions = tf.equal(tf.argmax(self.last_layer_sigmoid, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"))
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.last_layer, labels=self.labels))

            self.summary_accuracy = tf.summary.scalar("Accuracy", self.accuracy)
            self.summary_loss = tf.summary.scalar("Loss", self.loss)

            self.summaries = self.model.summaries
            self.summaries.append(self.summary_loss)
            self.summaries.append(self.summary_accuracy)

            self.summary = tf.summary.merge(self.summaries)

            self.optimizers = self.model.optimizers
            var_list = self.model.trainable_variables
            var_list.append(list(set(tf.trainable_variables()).symmetric_difference(set(temp))))
            self.optimizers.append(tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta).minimize(self.loss, var_list=var_list))

        self.saver = tf.train.Saver(max_to_keep=None)
        self.session = tf.Session()
        self.train_writer = tf.summary.FileWriter('log/training', self.session.graph)
        self.test_writer = tf.summary.FileWriter('log/testing', self.session.graph)
        tf.global_variables_initializer().run(session=self.session)

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
            samples_per_class = np.ceil([nsamples/len(i_)]).astype(np.int)[0]
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

    def read_single_models(self, data_file, ids):
        data = []
        labels = []
        x_ = np.load(data_file)['features']
        y_ = np.load(data_file)['targets']

        u_, i_ = np.unique(y_, return_index=True)
        samples_per_class = 1
        for i in range(len(i_)):
            if i in ids:
                for k in range(samples_per_class):
                    if i_[i] + k >= len(x_):
                        break
                    if i+1 < len(i_):
                        if i_[i] + k >= i_[i+1]:
                            break
                    data.append(np.reshape(x_[i_[i] + k], [32, 32, 32]))
                    labels.append([ids.index(i)])

        labels = np.asarray(labels)
        data = np.asarray(data)
        if data.shape[0] < self.batch_size:
            labels = np.repeat(labels, np.ceil(self.batch_size/labels.shape[0]), axis=0)
            data = np.repeat(data, np.ceil(self.batch_size / data.shape[0]), axis=0)

        labels = self.one_hot(labels, self.model.layers[len(self.model.layers)-1].output.shape.as_list()[1])
        data = data[..., np.newaxis].astype(np.float)
        return data, labels

    def read_next_chunk(self, data_file):
        data = []
        labels = []
        filedata = np.load(data_file)
        x_ = filedata['features']
        y_ = filedata['targets']
        filedata = []
        del filedata

        self.chunk_no += 1
        if self.chunk_no >= self.chunk_max:
            self.chunk_no = 0

        for i in range(self.chunk_no*self.chunk_size, np.minimum(len(x_), (self.chunk_no+1)*self.chunk_size)):
            data.append(np.reshape(x_[i], [32, 32, 32]))
            labels.append([y_[i]])

        x_ = []
        del x_
        y_ = []
        del y_

        labels = np.asarray(labels)
        data = np.asarray(data)
        if data.shape[0] < self.batch_size:
            labels = np.repeat(labels, np.ceil(self.batch_size/labels.shape[0]), axis=0)
            data = np.repeat(data, np.ceil(self.batch_size / data.shape[0]), axis=0)

        labels = self.one_hot(labels, self.model.layers[len(self.model.layers)-1].output.shape.as_list()[1])
        data = data[..., np.newaxis].astype(np.float)
        print('loaded %d chunk' % self.chunk_no)
        return data, labels

    def read_model_net_chunk(self, data_file, nsamples=10000):
        data = []
        labels = []
        filedata = np.load(data_file)
        x_ = filedata['features']
        y_ = filedata['targets']
        filedata=[]
        del filedata

        self.chunk_size = nsamples
        self.chunk_max = np.ceil(len(x_)*1.0/self.chunk_size).astype(np.int)
        self.chunk_no = 0

        for i in range(self.chunk_no*self.chunk_size, np.minimum(len(x_), (self.chunk_no+1)*self.chunk_size)):
            data.append(np.reshape(x_[i], [32, 32, 32]))
            labels.append([y_[i]])

        x_ = []
        del x_
        y_ = []
        del y_

        labels = np.asarray(labels)
        data = np.asarray(data)
        if data.shape[0] < self.batch_size:
            labels = np.repeat(labels, np.ceil(self.batch_size/labels.shape[0]), axis=0)
            data = np.repeat(data, np.ceil(self.batch_size / data.shape[0]), axis=0)

        labels = self.one_hot(labels, self.model.layers[len(self.model.layers)-1].output.shape.as_list()[1])
        data = data[..., np.newaxis].astype(np.float)
        print('loaded %d chunk'%self.chunk_no)
        return data, labels

    def read_next_chunk2(self):
        data = []
        labels = []

        self.chunk_no += 1
        if self.chunk_no >= self.chunk_max:
            self.chunk_no = 0

        for i in range(self.chunk_no*self.chunk_size, np.minimum(len(self.training_x_), (self.chunk_no+1)*self.chunk_size)):
            data.append(np.reshape(self.training_x_[i], [32, 32, 32]))
            labels.append([self.training_y_[i]])

        labels = np.asarray(labels)
        data = np.asarray(data)
        if data.shape[0] < self.batch_size:
            labels = np.repeat(labels, np.ceil(self.batch_size/labels.shape[0]), axis=0)
            data = np.repeat(data, np.ceil(self.batch_size / data.shape[0]), axis=0)

        labels = self.one_hot(labels, self.model.layers[len(self.model.layers)-1].output.shape.as_list()[1])
        data = data[..., np.newaxis].astype(np.float)
        print('loaded %d chunk' % self.chunk_no)
        return data, labels

    def read_model_net_chunk2(self, data_file, nsamples=1000):
        data = []
        labels = []
        x_ = np.load(data_file)['features']
        y_ = np.load(data_file)['targets']

        self.training_x_, self.training_y_ = self.shuffle_data(x_, y_)

        self.chunk_size = nsamples
        self.chunk_max = np.ceil(len(x_)*1.0/self.chunk_size).astype(np.int)
        self.chunk_no = 0

        for i in range(self.chunk_no*self.chunk_size, np.minimum(len(self.training_x_), (self.chunk_no+1)*self.chunk_size)):
            data.append(np.reshape(self.training_x_[i], [32, 32, 32]))
            labels.append([self.training_y_[i]])

        labels = np.asarray(labels)
        data = np.asarray(data)
        if data.shape[0] < self.batch_size:
            labels = np.repeat(labels, np.ceil(self.batch_size/labels.shape[0]), axis=0)
            data = np.repeat(data, np.ceil(self.batch_size / data.shape[0]), axis=0)

        labels = self.one_hot(labels, self.model.layers[len(self.model.layers)-1].output.shape.as_list()[1])
        data = data[..., np.newaxis].astype(np.float)
        print('loaded %d chunk'%self.chunk_no)
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

    def intensive_train(self, training_file, testing_file, nsamples=1000, ntsamples=100):

        if not os.path.exists('model'):
            os.makedirs('model')

        n_epochs = 10000
        training_x, training_l = self.read_single_models(training_file, [0, 8])
        testing_x, testing_l = self.read_single_models(training_file, [0, 8])
        # training_x, training_l = self.read_model_net(training_file, nsamples)
        # training_x, training_l = self.read_model_net_chunk(training_file)
        # testing_x, testing_l = self.read_model_net(testing_file, ntsamples)

        for epoch in range(n_epochs):

            # if epoch % 10 == 9:
            #     training_x, training_l = self.read_next_chunk(training_file)

            iters = np.floor([len(training_x)/ self.batch_size]).astype(np.int)[0]

            training_x, training_l = self.shuffle_data(training_x, training_l)
            testing_x, testing_l = self.shuffle_data(testing_x, testing_l)
            for iteration in range(iters):
                feed_dict = {self.input: training_x[iteration*self.batch_size:(iteration+1) * self.batch_size],
                             self.labels: training_l[iteration * self.batch_size:(iteration + 1) * self.batch_size]}
                feed_dict.update(self.model.get_feeding_dict())
                self.session.run(self.optimizers, feed_dict=feed_dict)
                self.session.run(self.model.optimizers, feed_dict=feed_dict)

            training_feed_dict = {self.input: training_x[:self.batch_size],
                                  self.labels: training_l[:self.batch_size]}
            training_feed_dict.update(self.model.get_feeding_dict())
            testing_feed_dict = {self.input: testing_x[:self.batch_size],
                                 self.labels: testing_l[:self.batch_size]}
            testing_feed_dict.update(self.model.get_feeding_dict())
            summary_training, loss, acc = self.session.run([self.summary, self.loss, self.accuracy],
                                                           feed_dict=training_feed_dict)
            summary_testing, loss2, acc2 = self.session.run([self.summary, self.loss, self.accuracy],
                                                            feed_dict=testing_feed_dict)
            print('Training epoch:', epoch, ', loss:', loss, ', accuracy:', acc)
            print('Testing epoch:', epoch, ', loss:', loss2, ', accuracy:', acc2)
            self.train_writer.add_summary(summary_training, epoch)
            self.train_writer.flush()

            self.test_writer.add_summary(summary_testing, epoch)
            self.test_writer.flush()

            if epoch % 50 == 10:
                self.saver.save(self.session, save_path='model/biasfree_' + str(epoch) + '.cptk')

        self.saver.save(self.session, save_path='model/last_model.cptk')

    def intensive_train_LAE(self, layer_no, starting_epoch, training_file, testing_file, nsamples=1000, ntsamples=100):
        assert 0 <= layer_no < len(self.model.layers)
        assert hasattr(self.model.layers[layer_no], 'optimizer')
        assert hasattr(self.model.layers[layer_no], 'l2_loss')
        assert hasattr(self.model.layers[layer_no], 'summary')
        assert hasattr(self.model.layers[layer_no], 'name')

        print('training ', self.model.layers[layer_no].name, ' (', layer_no, ') from epoch:', starting_epoch)

        if not os.path.exists('model'):
            os.makedirs('model')

        n_epochs = 10000
        training_x, training_l = self.read_model_net(training_file, nsamples)
        # training_x, training_l = self.read_model_net_chunk(training_file)
        testing_x, testing_l = self.read_model_net(testing_file, ntsamples)

        for epoch in range(starting_epoch + 1, n_epochs):

            # if epoch % 10 == 9:
            #     training_x, training_l = self.read_next_chunk(training_file)

            iters = np.floor([len(training_x)*0.2 / self.batch_size]).astype(np.int)[0]

            training_x, training_l = self.shuffle_data(training_x, training_l)
            testing_x, testing_l = self.shuffle_data(testing_x, testing_l)
            for iteration in range(iters):
                feed_dict = {self.input: training_x[iteration*self.batch_size:(iteration+1)*self.batch_size],
                             self.labels: training_l[iteration * self.batch_size:(iteration + 1) * self.batch_size]}
                feed_dict.update(self.model.get_feeding_dict())
                self.session.run(self.model.layers[layer_no].optimizer, feed_dict=feed_dict)

            training_feed_dict = {self.input: training_x[:self.batch_size],
                                  self.labels: training_l[:self.batch_size]}
            training_feed_dict.update(self.model.get_feeding_dict())
            testing_feed_dict = {self.input: testing_x[:self.batch_size],
                                 self.labels: testing_l[:self.batch_size]}
            testing_feed_dict.update(self.model.get_feeding_dict())

            summary_training, loss = self.session.run([self.model.layers[layer_no].summary, self.model.layers[layer_no].l2_loss],
                                                      feed_dict=training_feed_dict)
            summary_testing, loss2 = self.session.run([self.model.layers[layer_no].summary, self.model.layers[layer_no].l2_loss],
                                                      feed_dict=testing_feed_dict)

            print('Training epoch:', epoch, ', loss:', loss)
            print('Testing epoch:', epoch, ', loss:', loss2)
            self.train_writer.add_summary(summary_training, epoch)
            self.train_writer.flush()

            self.test_writer.add_summary(summary_testing, epoch)
            self.test_writer.flush()

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

    def get_code_at_layer(self, in_sample, layer_no):
        assert 0 <= layer_no < len(self.model.layers)
        assert hasattr(self.model.layers[layer_no], 'output')
        feed_dict = {self.input: np.repeat(in_sample, np.ceil(self.batch_size / in_sample.shape[0]), axis=0)[0:self.batch_size]}
        feed_dict.update(self.model.get_feeding_dict())
        return self.session.run(self.model.layers[layer_no].output, feed_dict=feed_dict)

    def reconstruct_code_at_layer(self, latent, layer_no):
        assert 0 <= layer_no < len(self.model.layers)
        assert hasattr(self.model.layers[1], 'deconv3_run')
        feed_dict = {self.input: np.zeros([self.batch_size, 32, 32, 32, 1])}
        feed_dict.update(self.model.get_feeding_dict(layer_no, latent))
        return self.session.run(self.model.layers[1].deconv3_run.output, feed_dict=feed_dict)

    def reconstruct_from_layer(self, in_sample, layer_no, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        latent = self.get_code_at_layer(in_sample=in_sample, layer_no=layer_no)
        out_sample = self.reconstruct_code_at_layer(latent=latent, layer_no=layer_no)
        self.save_sample(in_sample[0], save_folder + '/in.csv')
        self.save_sample(out_sample[0], save_folder + '/out.csv')
        copyfile('SampleGUI.exe', save_folder + '/SampleGUI.exe')
        avg_err = np.average(np.abs(in_sample[0]-out_sample[0]))
        print('Error:', avg_err)

    def reconstruct_from_layers(self, in_sample, layers_no, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.save_sample(in_sample[0], save_folder + '/in.csv')
        copyfile('SampleGUI.exe', save_folder + '/SampleGUI.exe')
        for layer_no in layers_no:
            latent = self.get_code_at_layer(in_sample=in_sample, layer_no=layer_no)
            out_sample = self.reconstruct_code_at_layer(latent=latent, layer_no=layer_no)
            self.save_sample(out_sample[0], save_folder + '/L%d.csv' % layer_no)
            avg_err = np.average(np.abs(in_sample[0]-out_sample[0]))
            print('Layer ', layer_no, ' Error:', avg_err)




# vox = VOXLAE(2)
# vox.load_model('model/biasfree_510.cptk')
# vox.intensive_train_LAE(layer_no=5,
#                         starting_epoch=360,
#                         training_file='data/ModelNet40/modelnet40_rot_train.npz',
#                         testing_file='data/ModelNet40/modelnet40_rot_test.npz')

# vox.intensive_train(training_file='data/ModelNet40/modelnet40_rot_train.npz',
#                     testing_file='data/ModelNet40/modelnet40_rot_test.npz')

# x, _ = vox.read_model_net('data/ModelNet40/modelnet40_rot_train.npz', 1)
# print(x.shape)

# vox.load_model('model/biasfree_310.cptk')
# vox.load_model('test_LAE/biasfree_260.cptk')
# for i in range(x.shape[0]):
#     vox.reconstruct_from_layers(in_sample=np.reshape(x[i], [1, 32, 32, 32, 1]), layers_no=[1, 2, 3, 4, 5], save_folder='test_LAE/full_loss_sigmoid/%d' % i)

# for i in range(1, 6):
#     latent_plane = vox.get_code_at_layer(in_sample=np.reshape(x[0], [1, 32, 32, 32, 1]), layer_no=i)
#     latent_chair = vox.get_code_at_layer(in_sample=np.reshape(x[8], [1, 32, 32, 32, 1]), layer_no=i)
#     reconstruct = vox.reconstruct_code_at_layer(latent=(latent_plane + latent_chair) / 2.0, layer_no=i)
#     vox.save_sample(reconstruct[0], 'test_LAE/plane_chair/L%d.csv' % i)
