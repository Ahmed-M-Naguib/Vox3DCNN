import tensorflow as tf
#
# def init_weights(shape, name):
#     return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
# def init_biases(shape, name):
#     return tf.Variable(name=name, initializer=tf.zeros(shape))
# def batchNorm(x, n_out, phase_train, scope='bn'):
#     with tf.variable_scope(scope):
#         beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
#         gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
#         batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
#         ema = tf.train.ExponentialMovingAverage(decay=0.5)
#
#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
#
#         mean, var = tf.cond(phase_train,
#                             mean_var_with_update,
#                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
#         normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#     return normed
# class batch_norm(object):
#     def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
#         with tf.variable_scope(name):
#             self.epsilon = epsilon
#         self.momentum = momentum
#         self.name = name
#     def __call__(self, x, train=True):
#         return tf.contrib.layers.batch_norm(x,
#                                             decay=self.momentum,
#                                             updates_collections=None,
#                                             epsilon=self.epsilon,
#                                             scale=True,
#                                             is_training=train,
#                                             scope=self.name)
# def threshold(x, val=0.5):
#     x = tf.clip_by_value(x, 0.5, 0.5001) - 0.5
#     x = tf.minimum(x * 10000, 1)
#     return x
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


class layer():
    def __init__(self):
        pass

class conv3D_layer(layer):
    def __init__(self,
                 last_layer,
                 shape,
                 strides,
                 name,
                 activation = lrelu,
                 normalizer=tf.contrib.layers.batch_norm,
                 padding="SAME",
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            self.weights = tf.get_variable('weights', shape=shape, initializer=initializer)
            self.conv3d = tf.nn.conv3d(last_layer.output, self.weights, strides=strides, padding=padding)
            if normalizer is None:
                self.batch_norm = self.conv3d
            else:
                self.batch_norm = normalizer(self.conv3d, is_training=trainable)
            if activation is None:
                self.output = self.batch_norm
            else:
                self.output = activation(self.batch_norm)

class maxpool_layer(layer):
    def __init__(self):
        pass

class fully_layer(layer):
    def __init__(self,
                 last_layer,
                 features,
                 name,
                 activation=tf.nn.sigmoid,
                 normalizer=None,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            in_size=1
            sizes=last_layer.output.shape.as_list()
            for i in range(1,len(sizes)):
                in_size *= sizes[i]
            print('size = ', in_size)
            self.weights = tf.get_variable(name='weights', shape=[in_size, features], initializer=initializer)
            self.biases  = tf.get_variable(name='biases', shape=[features], initializer=initializer)
            x = tf.reshape(last_layer.output, [-1, in_size])
            self.linear = tf.nn.bias_add(tf.matmul(x, self.weights), self.biases)
            if normalizer is None:
                self.batch_norm = self.linear
            else:
                self.batch_norm = normalizer(self.linear, is_training=trainable)
            if activation is None:
                self.output = self.batch_norm
            else:
                self.output = activation(self.batch_norm)
                if activation==tf.nn.sigmoid:
                    self.output = tf.maximum(tf.minimum(self.output, 0.99), 0.01)

class input_layer(layer):
    def __init__(self, shape, name):
        self.output = tf.placeholder(dtype=tf.float32, shape=shape, name=name)

class CNN3D():
    def __init__(self):
        self.layers=[]

    def add_layer(self, layer):
        self.layers.append(layer)
        return self.layers[len(self.layers)-1].output

    def add_input(self, shape, name):
        self.layers.append(input_layer(shape, name))
        return self.layers[len(self.layers) - 1].output

    def add_conv3d_layer(self,
                 shape,
                 strides,
                 name,
                 activation = lrelu,
                 normalizer=tf.contrib.layers.batch_norm,
                 padding="SAME",
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False):
        self.layers.append(conv3D_layer(self.layers[len(self.layers)-1],
                                        shape,
                                        strides,
                                        name,
                                        activation,
                                        normalizer,
                                        padding,
                                        initializer,
                                        trainable,
                                        reuse))
        return self.layers[len(self.layers) - 1].output

    def add_fully_layer(self,
                        features,
                        name,
                        activation=tf.nn.sigmoid,
                        normalizer=None,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        trainable=True,
                        reuse=False):
        self.layers.append(fully_layer(self.layers[len(self.layers)-1],
                                       features,
                                       name,
                                       activation,
                                       normalizer,
                                       initializer,
                                       trainable,
                                       reuse))
        return self.layers[len(self.layers) - 1].linear, self.layers[len(self.layers) - 1].batch_norm, self.layers[len(self.layers) - 1].output

    def compile_graph(self, log_dir, sess):
        self.train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        tf.global_variables_initializer().run(session=sess)

    def add_summary(self, summary, i):
        self.train_writer.add_summary(summary, i)