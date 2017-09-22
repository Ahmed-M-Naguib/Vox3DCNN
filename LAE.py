import tensorflow as tf


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


class Layer:
    def __init__(self):
        pass


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
                 beta=0.5
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
            self.output = self.conv3
            self.l2_loss = tf.reduce_sum((input_layer.output-self.deconv3.output)**2, range(1, len(input_layer.output.shape.as_list())))
            self.summary_loss = tf.summary.scalar(name + '/deconv3D/l2_loss', self.l2_loss)
            para_d = tf.trainable_variables() - temp
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta).minimize(self.l2_loss, var_list=para_d)

            self.selector_ph = None
            self.manual_input = None
            self.new_input = None
            self.deconv3_run = None

    def add_reconstruct(self, upper_layer):
        with tf.variable_scope(self.name, reuse=True):
            self.selector_ph = tf.placeholder(dtype=tf.bool, shape=[1])
            self.manual_input = InputLayer(shape=self.conv3.output.shape.as_list(), name='manual')
            self.new_input = tf.cond(self.selector_ph, self.manual_input, upper_layer)
            self.deconv3_run = Deconv3DLayer(last_layer=self.conv3,
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


class InputLayer(Layer):
    def __init__(self, shape, name):
        Layer.__init__(self)
        self.output = tf.placeholder(dtype=tf.float32, shape=shape, name=name)


class VariableInputLayer(Layer):
    def __init__(self, shape, name, initializer=tf.contrib.layers.xavier_initializer()):
        Layer.__init__(self)
        self.raw = tf.get_variable(name=name, shape=shape, initializer=initializer)
        self.output = tf.nn.sigmoid(self.raw)


class CNN3D:
    def __init__(self):
        self.layers = []

    def add_layer(self, alayer):
        if type(self.layers[-1]) is UpDownAE and not type(alayer) is UpDownAE:
            self.close_up_down_ae()
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
        for i in range(len(self.layers)-2, -1, -1):
            if type(self.layers[i]) is UpDownAE:
                self.layers[i].add_reconstruct(self.layers[i+1].deconv3_run)
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
                                beta=beta))
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
