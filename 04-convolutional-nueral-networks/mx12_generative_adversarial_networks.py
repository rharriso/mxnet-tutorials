""" How to make a genrative adversarial network """

from __future__ import print_function
import matplotlib as mpl
from matplotlib import pyplot
import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
import numpy

CTX = mx.cpu()

""" Generate some "real data" with normal dist """

RAND_NORM = nd.random_normal(shape=(1000, 2))
A_ARR = nd.array([[1, 2], [-0.1, 0.5]])
B_ARR = nd.array([1, 2])
X = nd.dot(RAND_NORM, A_ARR) + B_ARR
Y = nd.ones(shape=(1000, 1))

# create an iterator of "training data"
batch_size = 4
train_data = mx.io.NDArrayIter(X, Y, batch_size, shuffle=True)

pyplot.scatter(X[:, 0].asnumpy(), X[:, 1].asnumpy())
pyplot.show()
print("The covariance matrix is")
print(nd.dot(A_ARR, A_ARR.T))

# """ Defining a Network """

# build the generator
netG = nn.Sequential()
with netG.name_scope():
    netG.add(nn.Dense(2))

# build the descriminator (with 5 and 3 hidden units respectively)
netD = nn.Sequential()
with netD.name_scope():
    netD.add(nn.Dense(5, activation='tanh'))
    netD.add(nn.Dense(3, activation='tanh'))
    netD.add(nn.Dense(2))

# loss function
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# initialize the generator and discriminator
netG.initialize(mx.init.Normal(0.02), ctx=CTX)
netD.initialize(mx.init.Normal(0.02), ctx=CTX)

# """Setting up the training loop"""

real_label = mx.nd.ones((batch_size,), ctx=CTX)
generated_label = mx.nd.zeros((batch_size,), ctx=CTX)
metric = mx.metric.Accuracy()

# trainer for the generator and the discriminator
trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': 0.01})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': 0.05})

# set up logging
from datetime import datetime
import os
import time

# """Training Loop"""

stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')

for epoch in range(10):
    tic = time.time()
    train_data.reset()

    for i, batch in enumerate(train_data):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real_t
        data = batch.data[0].as_in_context(CTX)
        noise = nd.random_normal(shape=(batch_size, 2))

        with autograd.record():
            real_output = netD(data)
            errD_real = loss(real_output, real_label)
  
            generated = netG(noise)
            generated_output = netD(generated.detach())
            errD_generated = loss(generated_output, generated_label)
            errD = errD_real + errD_generated
  
            errD.backward()

        trainerD.step(batch_size)
        metric.update([real_label,], [real_output])
        metric.update([generated_label,], [generated_output])

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.record():
            output = netD(generated)
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(batch_size)

    name, acc = metric.get()
    metric.reset()
    print('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
    print('time: %f' % (time.time() - tic))
    noise = nd.random_normal(shape=(100, 2), ctx=CTX)
    generated = netG(noise)
    #pyplot.scatter(X[:, 0].asnumpy(),X[:,1].asnumpy())
    #pyplot.scatter(generated[:,0].asnumpy(),generated[:,1].asnumpy())
    #pyplot.title("Iteration " + `epoch`)
    #pyplot.show()

# Check it one last time        
noise = mx.nd.random_normal(shape=(100, 2), ctx=ctx)
generated = netG(noise)

pyplot.title("generated data on top")
plt.scatter(X[:, 0].asnumpy(),X[:,1].asnumpy())
plt.scatter(generated[:,0].asnumpy(),generated[:,1].asnumpy())
plt.show()

