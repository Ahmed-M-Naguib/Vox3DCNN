import tensorflow as tf
import numpy as np
from functools import reduce


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def mse(Y1, Y2):
    assert Y1.shape.as_list() == Y2.shape.as_list()
    return tf.reduce_sum(tf.pow(Y1 - Y2, 2)) / (reduce(lambda x, y: x * y, Y1.shape.as_list()))


class Layer:
    def __init__(self):
        pass


class TensorLayer(Layer):
    def __init__(self, tensor):
        Layer.__init__(self)
        self.output = tensor


class Deconv3DLayer(Layer):
    def __init__(self,
                 last_layer,
                 shape,
                 out_shape,
                 strides,
                 name,
                 activation=lrelu,
                 normalizer=tf.contrib.layers.batch_norm,
                 padding="SAME",
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False
                 ):
        Layer.__init__(self)
        temp = tf.trainable_variables()
        with tf.variable_scope(name, reuse=reuse):
            self.strides = strides
            self.padding = padding
            self.normalizer = normalizer
            self.activation = activation
            self.name = name
            self.out_shape = out_shape
            self.weights = tf.get_variable('weights', shape=shape, initializer=initializer)
            self.deconv3d = tf.nn.conv3d_transpose(last_layer.output, self.weights, strides=strides, padding=padding, output_shape=out_shape)
            if normalizer is None:
                    self.batch_norm = self.deconv3d
            else:
                    self.batch_norm = normalizer(self.deconv3d, is_training=trainable)
            if activation is None:
                self.output = self.batch_norm
            else:
                self.output = activation(self.batch_norm)
        if trainable:
            self.trainable_variables = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))
        else:
            self.trainable_variables = None


class Conv3DLayer(Layer):
    def __init__(self,
                 last_layer,
                 shape,
                 strides,
                 name,
                 activation=lrelu,
                 normalizer=tf.contrib.layers.batch_norm,
                 padding="SAME",
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False
                 ):
        Layer.__init__(self)
        temp = tf.trainable_variables()
        with tf.variable_scope(name, reuse=reuse):
            self.strides = strides
            self.padding = padding
            self.normalizer = normalizer
            self.activation = activation
            self.name = name
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
        if trainable:
            self.trainable_variables = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))
        else:
            self.trainable_variables = None


class UpDownAE(Layer):
    def __init__(self,
                 input_layer,
                 shape,
                 strides,
                 name,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 reuse=False,
                 up_activation=lrelu,
                 up_normalizer=tf.contrib.layers.batch_norm,
                 up_padding="SAME",
                 up_trainable=True,
                 down_activation=lrelu,
                 down_normalizer=tf.contrib.layers.batch_norm,
                 down_padding="SAME",
                 down_trainable=True,
                 lr=0.0001,
                 beta=0.5,
                 accum_loss=None
                 ):
        Layer.__init__(self)
        self.name = name
        self.out_shape = input_layer.output.shape.as_list()
        self.shape = shape
        self.strides = strides
        self.down_activation = down_activation
        self.down_padding = down_padding
        self.initializer = initializer
        self.down_trainable = down_trainable
        self.down_normalizer = down_normalizer
        temp = tf.trainable_variables()
        with tf.variable_scope(name, reuse=reuse):
            self.conv3 = Conv3DLayer(last_layer=input_layer,
                                     shape=shape,
                                     strides=strides,
                                     name='conv3D',
                                     activation=up_activation,
                                     normalizer=up_normalizer,
                                     padding=up_padding,
                                     initializer=initializer,
                                     trainable=up_trainable,
                                     reuse=reuse)
            if up_trainable:
                self.trainable_variables = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))
            else:
                self.trainable_variables = None
            temp = tf.trainable_variables()
            self.deconv3 = Deconv3DLayer(last_layer=self.conv3,
                                         out_shape=input_layer.output.shape.as_list(),
                                         shape=shape,
                                         strides=strides,
                                         name='deconv3D',
                                         activation=down_activation,
                                         normalizer=down_normalizer,
                                         padding=down_padding,
                                         initializer=initializer,
                                         trainable=down_trainable,
                                         reuse=reuse)
            self.output = self.conv3.output
            print('encoder size:', input_layer.output.shape.as_list())
            print('decoder size:', self.deconv3.output.shape.as_list())
            self.l2_loss = mse(input_layer.output, self.deconv3.output)
            # self.l2_loss = tf.reduce_mean(tf.reduce_sum((input_layer.output-self.deconv3.output)**2, np.arange(1, len(input_layer.output.shape.as_list()))))
            self.Accum_Loss = self.l2_loss
            if accum_loss is not None:
                self.Accum_Loss += accum_loss
            self.summary = [tf.summary.scalar(name + '/deconv3D/l2_loss', self.l2_loss), tf.summary.scalar(name + '/deconv3D/accum_loss', self.Accum_Loss)]
            para_d = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta).minimize(self.Accum_Loss, var_list=para_d)

            self.selector_ph = None
            self.manual_input = None
            self.new_input = None
            self.deconv3_run = None
            self.new_input_layer = None

    def add_reconstruct(self, upper_layer):
        with tf.variable_scope(self.name, reuse=True):
            self.selector_ph = tf.placeholder(dtype=tf.bool, shape=None, name='selector')
            self.manual_input = tf.placeholder(dtype=tf.float32, shape=self.conv3.output.shape.as_list(), name='manual')

            self.new_input = tf.cond(self.selector_ph, lambda: tf.add(self.manual_input, tf.zeros_like(self.manual_input)), lambda: tf.add(upper_layer, tf.zeros_like(upper_layer)))
            self.new_input_layer = TensorLayer(self.new_input)
            self.deconv3_run = Deconv3DLayer(last_layer=self.new_input_layer,
                                             out_shape=self.out_shape,
                                             shape=self.shape,
                                             strides=self.strides,
                                             name='deconv3D',
                                             activation=self.down_activation,
                                             normalizer=self.down_normalizer,
                                             padding=self.down_padding,
                                             initializer=self.initializer,
                                             trainable=self.down_trainable,
                                             reuse=True)


class MaxpoolLayer(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.trainable_variables = None
        pass


class FullyLayer(Layer):
    def __init__(self,
                 last_layer,
                 features,
                 name,
                 activation=tf.nn.sigmoid,
                 normalizer=None,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False):
        Layer.__init__(self)
        temp = tf.trainable_variables()
        with tf.variable_scope(name, reuse=reuse):
            in_size = 1
            sizes = last_layer.output.shape.as_list()
            for i in range(1, len(sizes)):
                in_size *= sizes[i]
            print('size = ', in_size)
            self.weights = tf.get_variable(name='weights', shape=[in_size, features], initializer=initializer)
            self.biases = tf.get_variable(name='biases', shape=[features], initializer=initializer)
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
                if activation == tf.nn.sigmoid:
                    self.output = tf.maximum(tf.minimum(self.output, 0.99), 0.01)
        if trainable:
            self.trainable_variables = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))
        else:
            self.trainable_variables = None


class InputLayer(Layer):
    def __init__(self, shape, name):
        Layer.__init__(self)
        self.output = tf.placeholder(dtype=tf.float32, shape=shape, name=name)
        self.trainable_variables = None


class VariableInputLayer(Layer):
    def __init__(self, shape, name, initializer=tf.contrib.layers.xavier_initializer()):
        Layer.__init__(self)
        temp = tf.trainable_variables()
        self.raw = tf.get_variable(name=name, shape=shape, initializer=initializer)
        self.output = tf.nn.sigmoid(self.raw)
        self.trainable_variables = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))


class CNN3D:
    def __init__(self):
        self.layers = []
        self.summaries = []
        self.optimizers = []
        self.LAE_dict = {}
        self.trainable_variables = []

    def add_layer(self, alayer):
        if len(self.layers) > 0:
            if type(self.layers[-1]) is UpDownAE and not type(alayer) is UpDownAE:
                self.close_up_down_ae()
        if hasattr(alayer, 'optimizer'):
            self.optimizers.append(alayer.optimizer)
        if hasattr(alayer, 'summary'):
            self.summaries.append(alayer.summary)
        if hasattr(alayer, 'trainable_variables'):
            if alayer.trainable_variables is not None:
                self.trainable_variables += alayer.trainable_variables
        self.layers.append(alayer)
        return self.layers[-1].output

    def add_variable_as_layer(self, variable):
        alayer = Layer()
        alayer.output = variable
        self.add_layer(alayer)
        return self.layers[-1].output

    def add_variable_input_layer(self, shape, name, initializer=tf.contrib.layers.xavier_initializer()):
        self.add_layer(VariableInputLayer(shape, name, initializer))
        return self.layers[-1].output, self.layers[-1].raw

    def add_input(self, shape, name):
        self.add_layer(InputLayer(shape, name))
        return self.layers[-1].output

    def close_up_down_ae(self):
        self.layers[-1].add_reconstruct(self.layers[-1].output)
        for i in range(len(self.layers)-1, -1, -1):
            if type(self.layers[i]) is UpDownAE:
                if i < len(self.layers)-1:
                    self.layers[i].add_reconstruct(self.layers[i+1].deconv3_run.output)
                self.LAE_dict.update({i: {'selector': self.layers[i].selector_ph, 'manual': self.layers[i].manual_input}})
            else:
                break

    def add_up_down_ae(self,
                       shape,
                       strides,
                       name,
                       initializer=tf.contrib.layers.xavier_initializer(),
                       reuse=False,
                       up_activation=lrelu,
                       up_normalizer=tf.contrib.layers.batch_norm,
                       up_padding="SAME",
                       up_trainable=True,
                       down_activation=lrelu,
                       down_normalizer=tf.contrib.layers.batch_norm,
                       down_padding="SAME",
                       down_trainable=True,
                       lr=0.0001,
                       beta=0.5
                       ):
        accum_loss = None
        if hasattr(self.layers[-1], 'Accum_Loss'):
            accum_loss = self.layers[-1].Accum_Loss
        self.add_layer(UpDownAE(input_layer=self.layers[-1],
                                shape=shape,
                                strides=strides,
                                name=name,
                                initializer=initializer,
                                reuse=reuse,
                                up_activation=up_activation,
                                up_normalizer=up_normalizer,
                                up_padding=up_padding,
                                up_trainable=up_trainable,
                                down_activation=down_activation,
                                down_normalizer=down_normalizer,
                                down_padding=down_padding,
                                down_trainable=down_trainable,
                                lr=lr,
                                beta=beta,
                                accum_loss=accum_loss))
        return self.layers[-1].output

    def add_conv3d_layer(self,
                         shape,
                         strides,
                         name,
                         activation=lrelu,
                         normalizer=tf.contrib.layers.batch_norm,
                         padding="SAME",
                         initializer=tf.contrib.layers.xavier_initializer(),
                         trainable=True,
                         reuse=False):
        self.add_layer(Conv3DLayer(self.layers[len(self.layers)-1],
                                   shape,
                                   strides,
                                   name,
                                   activation,
                                   normalizer,
                                   padding,
                                   initializer,
                                   trainable,
                                   reuse))
        return self.layers[-1].output

    def add_deconv3d_layer(self,
                           shape,
                           strides,
                           out_shape,
                           name,
                           activation=lrelu,
                           normalizer=tf.contrib.layers.batch_norm,
                           padding="SAME",
                           initializer=tf.contrib.layers.xavier_initializer(),
                           trainable=True,
                           reuse=False):
        self.add_layer(Deconv3DLayer(self.layers[len(self.layers)-1],
                                     shape,
                                     out_shape,
                                     strides,
                                     name,
                                     activation,
                                     normalizer,
                                     padding,
                                     initializer,
                                     trainable,
                                     reuse))
        return self.layers[-1].output

    def add_fully_layer(self,
                        features,
                        name,
                        activation=tf.nn.sigmoid,
                        normalizer=None,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        trainable=True,
                        reuse=False):
        self.add_layer(FullyLayer(self.layers[len(self.layers)-1],
                                  features,
                                  name,
                                  activation,
                                  normalizer,
                                  initializer,
                                  trainable,
                                  reuse))
        return self.layers[-1].linear, self.layers[-1].batch_norm, self.layers[-1].output

    def get_feeding_dict(self, layer_no=-1, injection_data=None):
        out_dict = {}
        selector_disp = ''
        if layer_no is -1:
            for k in self.LAE_dict:
                out_dict.update({self.LAE_dict[k]['selector']: False})
                out_dict.update({self.LAE_dict[k]['manual']: np.zeros(self.LAE_dict[k]['manual'].shape.as_list())})
                selector_disp += 'X'
        else:
            assert layer_no < len(self.layers)
            assert layer_no in self.LAE_dict
            assert list(injection_data.shape) == list(self.LAE_dict[layer_no]['manual'].shape.as_list()), "injection shape doesn't match. expected [%s], got [%s]" % (",".join(str(x) for x in self.LAE_dict[layer_no]['manual'].shape.as_list()),
                                                                                                                                                                      ",".join(str(x) for x in injection_data.shape))
            for k in self.LAE_dict:
                if k == layer_no:
                    out_dict.update({self.LAE_dict[k]['selector']: True})
                    out_dict.update({self.LAE_dict[k]['manual']: injection_data})
                    selector_disp += 'O'
                else:
                    out_dict.update({self.LAE_dict[k]['selector']: False})
                    out_dict.update({self.LAE_dict[k]['manual']: np.zeros(self.LAE_dict[k]['manual'].shape.as_list())})
                    selector_disp += 'X'
        print('selectors: ', selector_disp)
        return out_dict
