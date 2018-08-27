"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr_gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import plotting

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/local_home/tim/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/local_home/tim/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('-ed', '--energy_distance', dest='energy_distance', action='store_true', help='use energy distance in place of likelihood')
# optimization
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size during training per GPU')
parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('-ns', '--num_samples', type=int, default=1, help='How many batches of samples to output.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# energy distance or maximum likelihood?
if args.energy_distance:
    loss_fun = nn.energy_distance
else:
    loss_fun = nn.discretized_mix_logistic_loss

# initialize data loaders for train/test splits
if args.data_set == 'imagenet' and args.class_conditional:
    raise("We currently don't have labels for the small imagenet data set")
if args.data_set == 'cifar':
    import data.cifar10_data as cifar10_data
    DataLoader = cifar10_data.DataLoader
elif args.data_set == 'imagenet':
    import data.imagenet_data as imagenet_data
    DataLoader = imagenet_data.DataLoader
elif args.data_set == 'mydata':
    import data.my_data as mydata
    DataLoader = mydata.DataLoader
else:
    raise("unsupported dataset")
test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
obs_shape = test_data.get_observation_size() # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

# data place holders
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]

h_init = None
h_sample = [None] * args.nr_gpu
hs = h_sample

# create the model
model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance }
model = tf.make_template('model', model_spec)

# keep track of moving average
#all_params = tf.trainable_variables()
#ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
#maintain_averages_op = tf.group(ema.apply(all_params))
#ema_params = [ema.average(p) for p in all_params]

# get loss gradients over multiple GPUs + sampling
new_x_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # sample
        out = model(xs[i], h_sample[i], ema=None, dropout_p=0, **model_opt)
        if args.energy_distance:
            new_x_gen.append(out[0])
        else:
            new_x_gen.append(nn.sample_from_discretized_mix_logistic(out, args.nr_logistic_mix))

# sample from the model
def sample_from_model(sess):
    x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)

# init & save
saver = tf.train.Saver()

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    x = np.split(x, args.nr_gpu)
    feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
    return feed_dict

# //////////// perform training //////////////
with tf.Session() as sess:
    begin = time.time()

    # init
    ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)
    print('starting generation')

    # generate samples from the model
    sample_x = []
    for i in range(args.num_samples):
        sample_x.append(sample_from_model(sess))
    sample_x = np.concatenate(sample_x,axis=0)
    img_tile = plotting.img_tile(sample_x[:100], aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
    savepath = os.path.join(args.save_dir, 'generated')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plotting.plt.savefig(os.path.join(savepath,'%s_samples_generated.png' % (args.data_set)))
    plotting.plt.close('all')
    #np.savez(os.path.join(args.save_dir,'%s_sample%d.npz' % (args.data_set, epoch)), sample_x)

