import time
import mxnet as mx
import numpy as np
from matplotlib import pyplot as plt
from mxnet import gluon
from mxnet.gluon import nn

ctx = mx.gpu()
#ctx = mx.cpu()

#
# Helper functions
#
def visualize(img_data):
    """Add the image to the current plot"""
    plt.imshow(((img_data.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')

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
netG.initialize(mx.init.Normal(0.02), ctx=ctx)

#
# interpolate along a manifold
#
num_image = 12
latent_z_size = 100
latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
step = 0.05

for model in range(400, 801, 200):
    time.sleep(1)
    netG.load_params("mx13-models/generative-model-{}".format(model), ctx=ctx)
    time.sleep(1)
    num_image = 12
    latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
    step = 0.05
    for i in range(num_image):
        img = netG(latent_z)
        plt.subplot(3,4,i+1)
        visualize(img[0])
        latent_z += 0.05
    plt.suptitle('model: {}'.format(model))
    plt.show()
