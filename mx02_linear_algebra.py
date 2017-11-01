from mxnet import nd

x = nd.array([3.0])
y = nd.array([2.0])

# scalars
print('x + y = ', x + y)
print('x * y = ', x * y)
print('x / y = ', x / y)
print('x ** y = ', nd.power(x, y))

# arr to float
print(x.asscalar())

# vectors
u = nd.arange(4)
print('u = ', u)
print(u[3])
print('length: ', len(u)) 
print('shape: ', u.shape) 

# reshape array
x = nd.arange(20)
A = x.reshape((5, 4))
print('orig', x)
print('reshape', A)
print('norm(A)', nd.norm(A))

print('x.xT', nd.dot(x, x.T))
print('A.AT', nd.dot(A, A.T))

# tensors
tensor = x.reshape((5, 2, 2))
print('tensor = ', tensor)
print('sum(tensor)', nd.sum(tensor))
print('mean(tensor)', nd.mean(tensor))
print('norm(tensor)', nd.norm(tensor))