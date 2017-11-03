import mxnet as mx
from mxnet import nd, autograd

mx.random.seed(1)

x = nd.array([[1, 2], [3, 4]])
x.attach_grad()
with autograd.record():
    y = x * 2
    z = y * x

head_gradient = nd.array([[10, 1.], [.1, .01]])
z.backward(head_gradient)
print x.grad

a = nd.random_normal(shape=3)
a.attach_grad()
with autograd.record():
    b = a * 2
    while (nd.norm(b) < 1000).asscalar():
        b = b * 2
    
    if (mx.nd.sum(b) > 0).asscalar():
        c = b
    else:
        c = 100 * b

head_gradient = nd.array([0.01, 1.0, .1])
c.backward(head_gradient)
print(c)
print(a.grad)