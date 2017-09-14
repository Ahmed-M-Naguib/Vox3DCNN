import CNN3D
import tensorflow as tf
import numpy as np
import os
import shutil
from mayavi import mlab


class Vox3DCNN_model():
    def build_graph(self, n_objects, model, scope_name = 'Vox3DCNN', reuse=False, trainable=True):
        with tf.variable_scope(scope_name, reuse=reuse):
            for i in range(1, len(self.features)):
                model.add_conv3d_layer(
                    shape=[self.shapes[i], self.shapes[i], self.shapes[i], self.features[i - 1], self.features[i]],
                    strides=[1, self.strides[i], self.strides[i], self.strides[i], 1],
                    name='conv%d' % (i + 1),
                    activation=self.activation[i],
                    normalizer=self.normalizer[i],
                    padding=self.padding[i],
                    trainable=trainable,
                    reuse=reuse)

    def __init__(self, n_objects):

        self.model = CNN3D.CNN3D()
        self.batch_size = 32
        self.lr = 0.00001
        self.beta = 0.5
        self.cube_len = 64
        self.features = [1, 64, 128, 256, 512, 1024]
        self.shapes = np.repeat(4, len(self.features))
        self.strides = np.repeat(2, len(self.features))
        self.activation = np.append(np.repeat(CNN3D.lrelu, len(self.features) - 1), tf.nn.sigmoid)
        self.normalizer = np.append(np.repeat(tf.contrib.layers.batch_norm, len(self.features) - 1), None)
        self.padding = np.append(np.repeat("SAME", len(self.features) - 1), "VALID")
        self.input = self.model.add_input([self.batch_size, self.cube_len, self.cube_len, self.cube_len, 1], 'input')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, n_objects], name='labels')

        self.build_graph(n_objects=n_objects, model=self.model)

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

        if os.path.exists('log'):
            shutil.rmtree('log')
        os.makedirs('log')
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
            l = labels[idx]

            summary, loss, _ = self.session.run([self.summary, self.loss, self.optimizer], feed_dict={self.input: x, self.labels: l})
            print('Training epoch: ', epoch, ', loss:', loss)

            self.model.add_summary(summary, epoch)

            if epoch % 50 == 10:
                self.saver.save(self.session, save_path='model/biasfree_' + str(epoch) + '.cptk')
        self.saver.save(self.session, save_path='model/last_model.cptk')
    def save_model_dictionary(self, npz_file_name):
        for alayer in self.model.layers:
            pass

    def test_batch(self, data_dir, batch):
        batch = batch[0:32]
        return self.session.run(self.last_layer_sigmoid, feed_dict={self.input: batch})
    def test_sample(self, data_dir, sample):
        data = np.repeat(sample, np.ceil(self.batch_size / sample.shape[0]), axis=0)
        data = data[0:32]
        response = self.test_batch(data)
        return response[0]
    def test_accuracy(self, data_dir):
        volumes, labels = self.read_data(data_dir)
        volumes = volumes[0:32]
        labels = labels[0:32]
        volumes = volumes[..., np.newaxis].astype(np.float)
        return self.session.run([self.last_layer_sigmoid, self.accuracy, self.loss],
                                    feed_dict={self.input: volumes, self.labels: labels})
    def load_model(self, model):
        self.saver.restore(self.session, model)
    def build_reconstruction_graph(self, layer_number, address, filter_number):
        assert len(self.model.layers)>layer_number
        if len(self.model.layers[layer_number].output.shape.as_list())>=5:
            assert len(address) == 3
            assert self.model.layers[layer_number].output.shape.as_list()[1] > address[0]
            assert self.model.layers[layer_number].output.shape.as_list()[2] > address[1]
            assert self.model.layers[layer_number].output.shape.as_list()[3] > address[2]
            assert self.model.layers[layer_number].output.shape.as_list()[4] > filter_number
        else:
            assert self.model.layers[layer_number].output.shape.as_list()[1] > filter_number
        temp = set(tf.all_variables())

        n_objects = self.model.layers[len(self.model.layers) - 1].output.shape.as_list()[1]
        self.r_model = CNN3D.CNN3D()
        self.r_input, self.r_input_raw = self.r_model.add_variable_input_layer(shape=[self.batch_size, self.cube_len, self.cube_len, self.cube_len, 1], name='variable_input')
        self.build_graph(n_objects=n_objects, model=self.r_model, reuse=True)
        self.r_last_layer, _, self.r_last_layer_sigmoid = self.r_model.add_fully_layer(features=n_objects,
                                                                                 name='full%d' % (
                                                                                 len(self.features) - 1), reuse=True)
        if len(self.r_model.layers[layer_number].output.shape.as_list()) >= 5:
            self.r_activ = self.r_model.layers[layer_number].output[0, address[0], address[1], address[2], filter_number]
            print('connecting to 3DCNN layer')
        else:
            self.r_activ = - self.r_model.layers[layer_number].output[0, filter_number]
            print(self.r_model.layers[layer_number].output.shape.as_list())
            print('connecting to fully connected layer')
        self.r_summary_feature_activation = tf.summary.scalar("feature_activation", self.r_activ)
        self.session.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        tf.summary.FileWriter('log', self.session.graph)
        temp = set(tf.all_variables())
        self.r_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta).minimize(self.r_activ, var_list=[self.r_input_raw])
        self.session.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        tf.summary.FileWriter('log', self.session.graph)

    def save_sample(self, sample, path):
        assert sample.shape[0] == self.cube_len
        assert sample.shape[1] == self.cube_len
        assert sample.shape[2] == self.cube_len
        cntr=0
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
    def reconstruct_filter(self, layer_number, address, filter_number, number_iterations):
        self.build_reconstruction_graph(layer_number, address, filter_number)
        for iteration in range(number_iterations):
            r_summary, r_act, _ = self.session.run([self.r_summary_feature_activation, self.r_activ, self.r_optimizer])
            print('Iteration: ', iteration, ', activation:', r_act)
            self.model.add_summary(r_summary, iteration)
        r_sample, r_sample_raw = self.session.run([self.r_input, self.r_input_raw])
        r_sample = np.reshape(r_sample[0], [self.cube_len, self.cube_len, self.cube_len])
        r_sample_raw = np.reshape(r_sample_raw[0], [self.cube_len, self.cube_len, self.cube_len])
        self.save_sample(r_sample, 'reconstruction/sample.csv')
        self.save_sample(r_sample_raw, 'reconstruction/sample.csv')
        self.render3D(r_sample)
        self.render3D(r_sample_raw)
        return r_sample, r_sample_raw
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



# vox = Vox3DCNN_model(3)
# vox.load_model('model/last_model.cptk')
# vox.build_reconstruction_graph(6,None,0)
#vox.build_graph(n_objects=3, scope_name = 'newnet2', reuse=True)
# tf.summary.FileWriter('log', vox.session.graph)
# vox.train('data')
# vox.load_model('model/last_model.cptk')
# _, accuracy, _ = vox.test_accuracy('data')
# print('accuracy: ', accuracy)
# vox.reconstruct_filter( 6, None, 1, 5)