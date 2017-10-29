"""manipulating_mx_data"""

import mxnet as mx
from mxnet import nd, gpu

# set random see to always get the same result
mx.random.seed(1)

# just grab memory address and use them, don't initialize
unitialized_arr = nd.empty((3, 4))
print unitialized_arr

# initialize with zeros
zeros = nd.zeros((3, 4))
print zeros 

# sample from normal curve with a variance 1 and mean 0
normal_sample = nd.random_normal(0, 1, shape=(3, 4))
print normal_sample
print normal_sample.shape
print normal_sample.size

###
### Some Operations
###
# perform element wise addition an mul the two arrays together
print unitialized_arr + normal_sample
print unitialized_arr * normal_sample
# raise y to an exponent
print "\nnormal sample", normal_sample
print "\nexp normal", nd.exp(normal_sample)
print nd.dot(unitialized_arr, normal_sample.T)

#
# Using the GPU
#
ones = nd.ones((3, 4))
gpu_ones = ones.copyto(gpu(0))
normal_gpu = normal_sample.copyto(gpu(0))

print(nd.dot(normal_gpu, gpu_ones.T))
