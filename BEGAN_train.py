import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
import sys
import argparse
from ae_modules import Encoder, Decoder

tf.logging.set_verbosity(tf.logging.INFO)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='BEGAN test')
    parser.add_argument('--data_path',   dest='path',         default='fashion', type=str)
    parser.add_argument('--hidden',      dest='hidden',       default=32,        type=int,   help='hidden size')
    parser.add_argument('--conv-hidden', dest='conv_hidden',  default=64,        type=int,   help='conv hidden size')
    parser.add_argument('--gamma',       dest='gamma',        default=0.5,       type=float, help='gamma value for equilibrium')
    parser.add_argument('--lambda',      dest='lambda_',      default=0.001,     type=float, help='lambda for control')
    parser.add_argument('--lr',          dest='lr',           default=0.00001,   type=float, help='start learning rate')
    parser.add_argument('--batch',       dest='batch',        default=16,        type=int,   help='batch size')
    parser.add_argument('--iter',        dest='iter',         default=1000000,   type=int,   help='num of iteration')
    parser.add_argument('--gray',        dest='gray',         action='store_true',           help='gray or color')
    args = parser.parse_args()

    model_folder = "./models/BEGAN_{}_gray_{}_{}_{}_{}_{}".format(args.path,
                                                                  args.gray,
                                                                  args.lambda_,
                                                                  args.gamma,
                                                                  args.hidden,
                                                                  args.conv_hidden)
    # data load
    mnist = input_data.read_data_sets(args.path, one_hot=True)

    # config
    num_iter = args.iter
    B =        args.batch
    h =        args.hidden
    n =        args.conv_hidden
    gamma_ =   args.gamma

    gamma = tf.constant(gamma_, dtype=tf.float32)
    lambda_ = tf.Variable(args.lambda_,
                          dtype=tf.float32,
                          trainable=False)
    starter_learning_rate = args.lr
    
    k_initial = tf.constant(0, dtype=tf.float32)
    
    
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(starter_learning_rate, 
                                    global_step,
                                    100,
                                    0.98,
                                    staircase=True)
    lr_ = tf.Variable(lr,
                      dtype=tf.float32,
                      trainable=False)
    
    # placeholder
    # 여기는 mnist 처리...
    mnist_images = tf.placeholder(tf.float32, (None, 28*28), name="RealImage")

    if args.gray:
        im = tf.reshape(mnist_images, [-1, 28, 28, 1])
    else:
        im = tf.reshape(mnist_images, [-1, 28, 28])
        im = tf.stack([im, im, im],
                      axis=3)
    im = tf.image.resize_nearest_neighbor(im, [32, 32])
    real_images = 1 - im

    #real_images = tf.placeholder(tf.float32, (None, 32, 32, 3), name="RealImage")
    z = tf.placeholder(tf.float32, (None, h), name="z")
    k_prev = tf.placeholder(tf.float32, [])
    
    with tf.device("/gpu:0"):
        # Real
        latent_in_real, varE = Encoder(real_images, n, h, gray=args.gray)
        restored_real, varD = Decoder(latent_in_real, n, h, name="D", gray=args.gray)
        varDisc = varE + varD  # Discriminator의 variable을 가져와야지

        tf.summary.image("input_real",  real_images)
        tf.summary.image("output_real", restored_real)
        
        # fake
        fake_images, varGen = Decoder(z, n, h, name="G", gray=args.gray)
        latent_in_fake, _   = Encoder(fake_images, n, h, reuse=True, gray=args.gray)
        restored_fake, _    = Decoder(latent_in_fake, n, h, name="D", reuse=True, gray=args.gray)
 
        tf.summary.image("input_fake",  fake_images)
        tf.summary.image("output_fake", restored_fake)

        # real loss
        L_x =  tf.reduce_mean(tf.abs(real_images - restored_real))
        tf.summary.scalar("Real Loss", L_x)
        # fake loss
        L_z = tf.reduce_mean(tf.abs(fake_images - restored_fake))
        tf.summary.scalar("Fake Loss", L_z)

        # Discriminator/Generator loss
        L_D = L_x - k_prev * L_z
        L_G = L_z

        tf.summary.scalar("Discriminator Loss", L_D)
        tf.summary.scalar("Generator Loss", L_G)
        
        # control?
        k_next = k_prev + lambda_*(gamma*L_x - L_z)
        tf.summary.scalar("curr_K", k_prev)
        tf.summary.scalar("next_K", k_next)
    
        # convergence measure
        M_global = L_x + tf.abs(gamma*L_x - L_z)
        tf.summary.scalar("approx_convergence_measure", M_global)
        summary = tf.summary.merge_all()
        
        # gradient descent
        opt_D = tf.train.AdamOptimizer(lr_)
        opt_G = tf.train.AdamOptimizer(lr_)
        
        # 주의! : loss에 따라 gradient를 적용할 variable들이 다르다!!
        train_op_D = opt_D.minimize(L_D, var_list=varDisc)
        train_op_G = opt_G.minimize(L_G, global_step, var_list=varGen)
   
    saver = tf.train.Saver()
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(model_folder,
                                             sess.graph)
 
        k_t_ = sess.run(k_initial)
        t = tqdm(range(num_iter), desc="training BEGAN")
        for epoch in t:
            #### real_image ####
            batch_xs, _ = mnist.train.next_batch(B)
    
            #### fake_image ####
            Z = np.random.uniform(-1, 1, B * h)  # batch size가 달라도 된다.
            Z = np.reshape(Z, [B, h])
    
            r = sess.run([train_op_D,
                          train_op_G,
                          L_D,
                          L_G,
                          M_global,
                          k_next,
                          summary],
                         feed_dict={mnist_images: batch_xs,
                                    z:Z,
                                    k_prev: min(max(k_t_, 0), 1)})
            
            _, _, loss_D, loss_G, M_, k_t_, summary_ = r
            t.set_postfix(loss_D=loss_D, loss_G=loss_G, M_global=M_, k_t=k_t_)
            if np.isnan(k_t_):
                break
            if epoch % 1000 == 0:
                saver.save(sess, os.path.join(model_folder, "model.ckpt"))
                train_writer.add_summary(summary_, epoch)
    
        print("training done with {}-iteration.".format(num_iter), flush=True)
    
