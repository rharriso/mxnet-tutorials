import mxnet as mx
from mxnet import nd
import time

mx.random.seed(int(time.time()*1e6))

# multinomial sampling
probabilities = nd.ones(6) / 6
print(nd.sample_multinomial(probabilities))
print(nd.sample_multinomial(probabilities, shape=(10)))

# plot 1000 rolls
rolls = nd.sample_multinomial(probabilities, shape=(1000))
x = nd.arange(1000).reshape((1,1000)) + 1

# count of each roll after a given index
counts = nd.zeros((6, 1000))
totals = nd.zeros(6)
for i, roll in enumerate(rolls):
    totals[int(roll.asscalar())] += 1 # add one to the total
    counts[:, i] = totals # store totals as current vector

estimates = counts / x
print(estimates[:,0])
print(estimates[:,1])
print(estimates[:,100])

print(counts)

# plot the "estimate" for each possible roll
from matplotlib import pyplot as plt
plt.plot(estimates[0, :].asnumpy(), label="Estimated P(die=1)")
plt.plot(estimates[1, :].asnumpy(), label="Estimated P(die=2)")
plt.plot(estimates[2, :].asnumpy(), label="Estimated P(die=3)")
plt.plot(estimates[3, :].asnumpy(), label="Estimated P(die=4)")
plt.plot(estimates[4, :].asnumpy(), label="Estimated P(die=5)")
plt.plot(estimates[5, :].asnumpy(), label="Estimated P(die=6)")
plt.axhline(y=0.16666, color='black', linestyle='dashed', label='0.1666')
plt.legend()
plt.show()

# Naive Base classification
import numpy as np

# go over one obsercation at a time (not going for speed)
def transform(data, label):
    return data.astype(np.float32) / 255, label.astype(np.float32)

mnist_train = mx.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = mx.gluon.data.vision.MNIST(train=False, transform=transform)

# initialize the count stats for p(y) and p(x_i|y)
# init all with count one, avoid division by zero (Laplace Smoothing)
ycount = nd.ones(shape=(10))
xcount = nd.ones(shape=(784, 10))

# add up how many times a pixel is on for each image
for data, label in mnist_train:
    x = data.reshape((784,))
    y = int(label)
    ycount[y] += 1
    xcount[:, y] += x # add the normalized binary image to the count

# nomalize the probability (divide by total count)
# average the pixel is on per image
for i in range(10):
    xcount[:, i] = xcount[:, i] / ycount[i]

# likewise, compute the probability p(y)
py = ycount / nd.sum(ycount)

# plot probability of pixel per label (should look like the number)
import matplotlib.pyplot as plt
fig, figarr = plt.subplots(1, 10, figsize=(15, 15))
for i in range(10):
    figarr[i].imshow(xcount[:, i].reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[i].get_yaxis().set_visible(False)
    figarr[i].get_xaxis().set_visible(False)
plt.show()

logxcount = nd.log(xcount)
logxcountneg = nd.log(1-xcount)
logpy = nd.log(py)

fig, figarr = plt.subplots(2, 10, figsize=(15, 3))

# show 10 images
ctr = 0
for data, label in mnist_test:
    x = data.reshape((784,))
    y = int(label)

    # we need to incorporate the prior probability p(y) since p(y|x) is
    # proportional to p(x|y) p(y)
    logpx = logpy.copy()
    for i in range(10):
        # compute the log probability for a digit
        logpx[i] = nd.dot(logxcount[:, i], x) + nd.dot(logxcountneg[:, i], 1-x)
    # normalize to prevent overflow or underflow by subtracting the largest
    # value
    logpx -= nd.max(logpx)
    # and compute the softmax using logpx
    px = nd.exp(logpx).asnumpy()
    px /= np.sum(px)

    # bar chart and image of digit
    figarr[1, ctr].bar(range(10), px)
    figarr[1, ctr].axes.get_yaxis().set_visible(False)
    figarr[0, ctr].imshow(x.reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[0, ctr].axes.get_xaxis().set_visible(False)
    figarr[0, ctr].axes.get_yaxis().set_visible(False)
    ctr += 1
    if ctr == 10:
        break

plt.show()
