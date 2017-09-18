import tensorflow as tf

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

class layer():
    def __init__(self):
        pass
    def override_input(self):
        pass

class deconv3D_layer(layer):
    def __init__(self,
                 last_layer,
                 shape,
                 out_shape,
                 strides,
                 name,
                 activation = lrelu,
                 normalizer=tf.contrib.layers.batch_norm,
                 padding="SAME",
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False
                 ):
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
                 reuse=False
                 ):
        with tf.variable_scope(name, reuse=reuse):
            self.strides=strides
            self.padding=padding
            self.normalizer=normalizer
            self.activation=activation
            self.name=name
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
class variable_input_layer(layer):
    def __init__(self, shape, name, initializer=tf.contrib.layers.xavier_initializer()):
        self.raw = tf.get_variable(name=name, shape=shape, initializer=initializer)
        self.output = tf.nn.sigmoid(self.raw)

class CNN3D():
    def __init__(self):
        self.layers=[]
    def add_layer(self, alayer):
        self.layers.append(alayer)
        return self.layers[len(self.layers)-1].output
    def add_variable_as_layer(self, variable):
        alayer = layer()
        alayer.output = variable
        self.layers.append(alayer)
        return self.layers[len(self.layers)-1].output
    def add_variable_input_layer(self, shape, name, initializer=tf.contrib.layers.xavier_initializer()):
        self.layers.append(variable_input_layer(shape, name, initializer))
        return self.layers[len(self.layers)-1].output, self.layers[len(self.layers)-1].raw
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
        self.layers.append(deconv3D_layer(self.layers[len(self.layers)-1],
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