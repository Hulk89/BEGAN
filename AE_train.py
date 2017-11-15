import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm
import os
from tensorflow.examples.tutorials.mnist import input_data
from ae_modules import Encoder, Decoder


tf.logging.set_verbosity(tf.logging.INFO)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='BEGAN test')
    parser.add_argument('--data_path',   dest='path',         default='fashion', type=str)
    parser.add_argument('--hidden',      dest='hidden',       default=32,        type=int,   help='hidden size')
    parser.add_argument('--gray',        dest='gray',         action='store_true',           help='gray or color')
    parser.add_argument('--conv-hidden', dest='conv_hidden',  default=64,        type=int,   help='conv hidden size')
    parser.add_argument('--lr',          dest='lr',           default=0.00001,   type=float, help='start learning rate')
    parser.add_argument('--batch',       dest='batch',        default=16,        type=int,   help='batch size')
    parser.add_argument('--iter',        dest='iter',         default=1000,      type=int,   help='num of iteration')
    parser.add_argument('--normalize',   dest='norm',         action="store_true",           help='layernorm and residual')
    parser.add_argument("--prefix",      dest='pre',          default='')
    args = parser.parse_args()
    print("Settings: {}".format(args))
    model_folder = "./models/{}_AE_{}_gray_{}_normalize_{}_{}_{}".format(args.pre,
                                                    args.path,
                                                    args.gray,
                                                    args.norm,
                                                    args.hidden,
                                                    args.conv_hidden)
 
  
    B = args.batch
    num_iter = args.iter

    mnist = input_data.read_data_sets(args.path, one_hot=True)

    Hidden = args.hidden
    conv_Hidden = args.conv_hidden
    
    global_step = tf.Variable(0, trainable=False)
    
    starter_learning_rate = args.lr
    lr = tf.train.exponential_decay(starter_learning_rate, 
                                    global_step,
                                    100,
                                    0.5,
                                    staircase=True)
    
    lr_ = tf.Variable(lr,
                      dtype=tf.float32,
                      trainable=False)
    # placeholder
    mnist_images = tf.placeholder(tf.float32, (None, 28*28), name="RealImage")
    if args.gray:
        im = tf.reshape(mnist_images, [-1, 28, 28, 1])
    else:
        im = tf.reshape(mnist_images, [-1, 28, 28])
        im = tf.stack([im, im, im],
                      axis=3)
    im = tf.image.resize_nearest_neighbor(im, [32, 32])
    real_images = 1 - im

    with tf.device("/gpu:0"):
        # Real
        latent_in_real, _ = Encoder(real_images,
                                    conv_Hidden,
                                    Hidden,
                                    normalize=args.norm,
                                    gray=args.gray)
        restored_real, _ = Decoder(latent_in_real,
                                   conv_Hidden,
                                   Hidden,
                                   name="D",
                                   normalize=args.norm,
                                   gray=args.gray)
        tf.summary.image("input",  real_images)
        tf.summary.image("output", restored_real)
        # real loss
        L_x =  tf.reduce_mean(tf.abs(real_images - restored_real))
        tf.summary.scalar("loss", L_x)
        merged_summary = tf.summary.merge_all()

        # gradient descent
        opt_D = tf.train.AdamOptimizer(lr_)

        train_op_D = opt_D.minimize(L_x, global_step=global_step)


    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(model_folder,
                                             sess.graph)

        t = tqdm(range(num_iter), desc="training AE")
        for epoch in t:
            #### real_image ####
            batch_xs, _ = mnist.train.next_batch(B)
            _, loss_D, summary = sess.run([train_op_D,
                                           L_x,
                                           merged_summary],
                                          feed_dict={mnist_images: batch_xs})
            t.set_postfix(loss_D=loss_D)

            if epoch % 1000 == 0:
                saver.save(sess, os.path.join(model_folder, "/model.ckpt"))
                train_writer.add_summary(summary, epoch)

