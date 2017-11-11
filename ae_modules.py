import tensorflow as tf
import numpy as np

def Encoder(images,  ## N x 32 x 32 x (3 or 1)
            n,
            h,
            reuse=False,
            stddev=0.02,
            normalize=True,
            gray=False):
    input_ = images
    if gray is True:
        channel = 1
    else:
        channel = 3
    filter_shapes = [[3, 3, channel, n],
                     [3, 3, n, n],
                     [3, 3, n, 2*n],
                     [3, 3, 2*n, 2*n],
                     [3, 3, 2*n, 3*n],
                     [3, 3, 3*n, 3*n],
                     [3, 3, 3*n, 3*n]]

    subsampling_layer = [2, 4]
    subsampling_size = [[16, 16],[8, 8]]

    with tf.variable_scope("encoder", reuse=reuse) as scope:
        # Convolution
        for i, filter_shape in enumerate(filter_shapes):
            conv_weight = tf.get_variable("conv_weight_{}".format(i),
                                          filter_shape,
                                          initializer=tf.truncated_normal_initializer(stddev=stddev))
            res = tf.nn.conv2d(input_,
                               conv_weight,
                               [1, 1, 1, 1],
                               "SAME")
            if normalize:
                res = tf.contrib.layers.layer_norm(res)

            res = tf.nn.elu(res)

            if normalize and (tf.shape(input_) == tf.shape(res)):
                res = input_ + res

            if i in subsampling_layer:  # subsampling
                res = tf.image.resize_nearest_neighbor(res,
                                                       subsampling_size[subsampling_layer.index(i)])
            input_ = res
        '''
        16 x 8*8*3*n
        '''
        before_fnn = tf.contrib.layers.flatten(input_)
        '''
        16 x h
        '''
        after_fnn = tf.layers.dense(before_fnn, h)

        variables = tf.contrib.framework.get_variables(scope)

        return after_fnn, variables

def Decoder(encoded,  ## N x h
            n,
            h,
            name="D",
            reuse=False,
            stddev=0.02,
            normalize=True,
            gray=False):
    if gray is True:
        channel = 1
    else:
        channel = 3
    filter_shapes = [[3, 3, n, n],
                     [3, 3, n, n],
                     [3, 3, n, n],  # 2*n, n인데 channel 뻥튀기가 귀찮아서.. 일단..
                     [3, 3, n, n],
                     [3, 3, n, n],  # 역시 같은 이유
                     [3, 3, n, n],
                     [3, 3, n, channel]]

    upscaling_layer = [1, 3]
    upscaling_size = [[16, 16],[32, 32]]

    with tf.variable_scope("{}_decoder".format(name), reuse=reuse) as scope:
        # fnn
        fnn_res = tf.layers.dense(encoded, 8 * 8 * n)
        fnn_res = tf.nn.elu(fnn_res)
        input_ = tf.reshape(fnn_res, [-1, 8, 8, n])
        # Convolution
        for i, filter_shape in enumerate(filter_shapes):
            conv_weight = tf.get_variable("conv_weight_{}".format(i),
                                          filter_shape,
                                          initializer=tf.truncated_normal_initializer(stddev=stddev))
            res = tf.nn.conv2d(input_,
                               conv_weight,
                               [1, 1, 1, 1],
                               "SAME")

            if i != len(filter_shapes) -1:  # 마지막에 elu가 끼면 안될 듯...
                if normalize:
                    res = tf.contrib.layers.layer_norm(res)
                res = tf.nn.elu(res)

            if i in upscaling_layer:  # subsampling
                res = tf.image.resize_nearest_neighbor(res,
                                                       upscaling_size[upscaling_layer.index(i)])
            if normalize:
                if tf.shape(input_) == tf.shape(res):
                    res = input_ + res

            input_ = res

        res = tf.sigmoid(res)  ## 제일 중요....

        variables = tf.contrib.framework.get_variables(scope)

        return res, variables

