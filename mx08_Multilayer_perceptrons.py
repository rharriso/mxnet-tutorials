"""Multilayer perceptrons with gluon"""

from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon

ctx = mx.gpu()

batch_size = 64
num_inputs = 784
num_outputs = 10 

def transform(data_tran, label_tran):
    return data_tran.astype(np.float32) / 255, label_tran.astype(np.float32)

train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(train=True,transform=transform),
    batch_size=batch_size,
    shuffle=False
)
test_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(train=False ,transform=transform),
    batch_size=batch_size,
    shuffle=False
)

num_hidden = 256
net = gluon.nn.Sequential()

with net.name_scope():
    net.add(gluon.nn.Dense(num_hidden, activation='relu'))
    net.add(gluon.nn.Dense(num_hidden, activation='relu'))
    net.add(gluon.nn.Dense(num_outputs))

# parameter initialization
net.collect_params().initialize(
    mx.init.Xavier(magnitude=2.24), ctx=ctx
)

# softmax coss-entropy loss
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# create trainer with params and 
trainer = gluon.Trainer(
    net.collect_params(),
    'sgd',
    { 'learning_rate': 0.1 }
)

# evaluation function
def evaluation_fn(data_iterator, net_eval):
    acc = mx.metric.Accuracy()
    for _, (data_eval, label_eval) in enumerate(data_iterator):
        data_eval = data_eval.as_in_context(ctx).reshape((-1, 784))
        label_eval = label_eval.as_in_context(ctx)
        net_output = net_eval(data_eval)
        predictions = nd.argmax(net_output, axis=1)
        acc.update(preds=predictions, labels=label_eval)
    return acc.get()[1]

epochs = 10
smoothing_constant = 0.1

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)

        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
            loss.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if (i == 0 and e == 0) else
                       (1 - smoothing_constant) * moving_loss +
                       (smoothing_constant) * curr_loss)
    
    test_accuracy = evaluation_fn(test_data, net)
    train_accuracy = evaluation_fn(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, moving_loss, train_accuracy, test_accuracy))