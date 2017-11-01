from __future__ import print_function
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import time

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet import autograd
import numpy as np

epochs = 1500  # set dow by default for tests, set higher when you actual run
batch_size = 64
image_width = 64
image_height = 64
latent_z_size = 100

ctx = mx.gpu()

learning_rate = 0.0002
beta1 = 0.5


#
# Helper functions
#

def visualize(img_data):
    """Add the image to the current plot"""
    plt.imshow(((img_data.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')


def transform(data):
    """transform images into 64x64"""
    # resize
    data = mx.image.imresize(data, image_width, image_height)
    # transpose dimentions (w, h, 3) -> (3, w, h)
    data = nd.transpose(data, (2, 0, 1))
    # normalize from [0, 256] to [-1, 1]
    data = data.astype(np.float32) / 127.5 - 1
    # if greyscale tripple the values
    if data.shape[0] == 1:
        data = nd.tile(data, (3, 1, 1))
    # wtf does this do?
    return data.reshape((1,) + data.shape)

# end transform


# downloaded from 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
data_path = os.path.expanduser("~/Desktop/lfw-deepfunneled")
img_list = []

print("resizing images")

# TODO: refactor this in a map/filer function
# TODO: batch this so we don't run out of memory
# create training data from the images
for path, _, fnames in os.walk(data_path):
    for fname in fnames:
        # 6000 is the most my computer can handle
        if len(img_list) > 6000 or not fname.endswith('.jpg'):
            continue
        img = os.path.join(path, fname)
        img_list.append(transform(mx.image.imread(img)))

# consolidate training set
train_data = mx.io.NDArrayIter(
  data=nd.concatenate(img_list)
  , batch_size=batch_size
)

# visualize images
#for i in range(4):
#    plt.subplot(1, 4, i + 1)
#    visualize(img_list[i + 10][0])
#plt.show()

print("building generator and descriminator")

# build the generator
nc = 3
ngf = 64
netG = nn.Sequential()
with netG.name_scope():
    # input is Z, going into a convolution
    netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 4 x 4
    netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 8 x 8
    netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 16 x 16
    netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 32 x 32
    netG.add(nn.Conv2DTranspose(nc, 4, 2, 1, use_bias=False))
    netG.add(nn.Activation('tanh'))
    # state size. (nc) x 64 x 64

# build the discriminator
ndf = 64
netD = nn.Sequential()
with netD.name_scope():
    # input is (nc) x 64 x 64
    netD.add(nn.Conv2D(ndf, 4, 2, 1, use_bias=False))
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 32 x 32
    netD.add(nn.Conv2D(ndf * 2, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 16 x 16
    netD.add(nn.Conv2D(ndf * 4, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 8 x 8
    netD.add(nn.Conv2D(ndf * 8, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 4 x 4
    netD.add(nn.Conv2D(1, 4, 1, 0, use_bias=False))

#
# Set up loss and optimizer
#
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# init gen and discrim with normal sample
#netG.initialize(mx.init.Normal(0.02), ctx=ctx)
netG.load_params("mx13-models/generative-model-500", ctx=ctx)
#netD.initialize(mx.init.Normal(0.02), ctx=ctx)
netD.load_params("mx13-models/descriminative-model500", ctx=ctx)

# trainer for the gen and discrim
trainerG = gluon.Trainer(netG.collect_params(), 'adam', {
    'learning_rate': learning_rate,
    'beta1': beta1
})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {
    'learning_rate': learning_rate,
    'beta1': beta1
})

#
# Training
# Loop
#

from datetime import datetime
import time
import logging

real_label = nd.ones((batch_size,), ctx=ctx)
fake_label = nd.zeros((batch_size,),ctx=ctx)

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()
metric = mx.metric.CustomMetric(facc)

stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')
logging.basicConfig(level=logging.DEBUG)

for epoch in range(500, epochs):
    tic = time.time()
    btic = time.time()
    train_data.reset()
    i = 0
    for batch in train_data:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        data = batch.data[0].as_in_context(ctx)
        latent_z = mx.nd.random_normal(0, 1, shape=(batch_size, latent_z_size, 1, 1), ctx=ctx)

        with autograd.record():
            # train with real image
            output = netD(data).reshape((-1, 1))
            errD_real = loss(output, real_label)
            metric.update([real_label,], [output,])

            # train with fake image
            fake = netG(latent_z)
            output = netD(fake).reshape((-1, 1))
            errD_fake = loss(output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            metric.update([fake_label,], [output,])

        trainerD.step(batch.data[0].shape[0])

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.record():
            fake = netG(latent_z)
            output = netD(fake).reshape((-1, 1))
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(batch.data[0].shape[0])

        # Print log infomation every ten batches
        if i % 10 == 0:
            name, acc = metric.get()
            logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
            logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f at i %d epoch %d'
                     %(nd.mean(errD).asscalar(),
                       nd.mean(errG).asscalar(), acc, i, epoch))
        i = i + 1
        btic = time.time()

    #
    # save model after epoch
    # 
    if  0 == epoch or epoch % 50 == 0:
        logging.info('Saving models at epoch: {}'.format(epoch))
        netG.save_params("mx13-models/generative-model-" + `epoch`)
        netD.save_params("mx13-models/descriminative-model" + `epoch`)

    name, acc = metric.get()
    metric.reset()

#
# display some results after training
#
num_image = 8
for i in range(num_image):
    latent_z = mx.nd.random_normal(0, 1,
      shape=(1, latent_z_size, 1, 1),
      ctx=ctx
    )
    img = netG(latent_z)
    plt.subplot(2,4,i+1)
    visualize(img[0])
plt.show()


#
# interpolate along a manifold
#
num_image = 12
latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
step = 0.05
for i in range(num_image):
    img = netG(latent_z)
    plt.subplot(3,4,i+1)
    visualize(img[0])
    latent_z += 0.05
plt.show()

