# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:12:04 2018

@author: feywell
"""

from __future__ import print_function
import mxnet as mx
from mxnet import nd,init,gluon,autograd
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.data.vision import transforms
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils import *
## 加载数据

class DataLoader(object):
    """similiar to gluon.data.DataLoader, but might be faster.
    The main difference this data loader tries to read more exmaples each
    time. But the limits are 1) all examples in dataset have the same shape, 2)
    data transfomer needs to process multiple examples at each time
    """
    def __init__(self, dataset, batch_size, shuffle, transform=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform

    def __iter__(self):
        data = self.dataset[:]
        X = data[0]
        y = nd.array(data[1])
        n = X.shape[0]
        if self.shuffle:
            idx = np.arange(n)
            np.random.shuffle(idx)
            X = nd.array(X.asnumpy()[idx])
            y = nd.array(y.asnumpy()[idx])

        for i in range(n//self.batch_size):
            if self.transform is not None:
                yield self.transform(X[i*self.batch_size:(i+1)*self.batch_size], 
                                     y[i*self.batch_size:(i+1)*self.batch_size])
            else:
                yield (X[i*self.batch_size:(i+1)*self.batch_size],
                       y[i*self.batch_size:(i+1)*self.batch_size])

    def __len__(self):
        return len(self.dataset)//self.batch_size

def load_data_mnist(batch_size, resize=None, root="~/.mxnet/datasets/mnist"):
    """download the fashion mnist dataest and then load into memory"""
    def transform_mnist(data, label):
        # Transform a batch of examples.
        if resize:
            n = data.shape[0]
            new_data = mx.nd.zeros((n, resize, resize, data.shape[3]))
            for i in range(n):
                new_data[i] = mx.image.imresize(data[i], resize, resize)
            data = new_data
        # change data from batch x height x width x channel to batch x channel x height x width
        return mx.nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')

    mnist_train = gluon.data.vision.MNIST(root=root, train=True, transform=None)
    mnist_test = gluon.data.vision.MNIST(root=root, train=False, transform=None)
    # Transform later to avoid memory explosion. 
    train_data = DataLoader(mnist_train, batch_size, shuffle=True, transform=transform_mnist)
    test_data = DataLoader(mnist_test, batch_size, shuffle=False, transform=transform_mnist)
    return (train_data, test_data)
batch_size=64
train_data, test_data = load_data_mnist(batch_size)


# ## 训练数据集
#train_data = DataLoader(
#         vision.datasets.MNIST(train=True, 
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),batch_size=2, shuffle=False
#               )
#
#print('train_data:',type(train_data))
# ## 测试数据集
#test_data = DataLoader(
#         vision.datasets.MNIST(train=False,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),batch_size=2, shuffle=False
#               )

#train_data = DataLoader(
#			vision.datasets.MNIST(train=True, transform=lambda data, label: (data.astype(np.float32)/255, label)), 
#			batch_size=2, shuffle=True
#			)
#
#test_data = DataLoader(
#			vision.datasets.MNIST(train=False, transform=lambda data, label: (data.astype(np.float32)/255, label)), 
#			batch_size=2, shuffle=False
#			)
print('test_data:',type(train_data))

print('CREATE class STN')
class STN(nn.HybridBlock):
    def __init__(self):
        super(STN, self).__init__()
        with self.name_scope():

        # Spatial transformer localization-network
            loc = self.localization = nn.HybridSequential() 
            loc.add(nn.Conv2D(8, kernel_size=7))
            loc.add(nn.MaxPool2D(strides=2))
            loc.add(nn.Activation(activation='relu'))
            loc.add(nn.Conv2D(10, kernel_size=5))
            loc.add(nn.MaxPool2D(strides=2))
            loc.add(nn.Activation(activation='relu'))
            
            # Regressor for the 3 * 2 affine matrix
            fc_loc = self.fc_loc = nn.HybridSequential()
            fc_loc.add(nn.Dense(32,activation='relu'))
            fc_loc.add(nn.Dense(3 * 2,weight_initializer='zeros'))
            
    # Spatial transformer network forward function            
    def hybrid_forward(self,F, x):    
        xs = self.localization(x)
        xs = xs.reshape((-1, 10 * 3 * 3))
        theta = self.fc_loc(xs)
        theta = theta.reshape((-1, 2*3))

        grid = F.GridGenerator(data=theta, transform_type='affine',target_shape=(28,28),name='grid')

        x = F.BilinearSampler(data=x,grid=grid,name='sampler' )

        return x        

print('CREATE class NET')
class Net(nn.HybridBlock):
    def __init__(self):
        super(Net, self).__init__()
        with self.name_scope():
            self.model = nn.HybridSequential()
            self.model.add(STN())
            self.model.add(nn.Conv2D(10, kernel_size=5))
            self.model.add(nn.MaxPool2D())
            self.model.add(nn.Activation(activation='relu'))
            self.model.add(nn.Conv2D(20, kernel_size=5))
            self.model.add(nn.Dropout(.5))
            self.model.add(nn.MaxPool2D())
            self.model.add(nn.Activation(activation='relu'))
            self.model.add(nn.Flatten())
            self.model.add(nn.Dense(50))
            self.model.add(nn.Activation(activation='relu'))
            self.model.add(nn.Dropout(.5))
            self.model.add(nn.Dense(10))

    def hybrid_forward(self,F, x):
        # transform the input
#        x = STN(x)
        for i,b in enumerate(self.model):
            x = b(x)

        return x

ctx = mx.gpu(0)    
net = Net()
print('net:',net)
net.initialize(ctx=ctx, init=init.Xavier())

w = net.model[0].fc_loc[1].weight
b = net.model[0].fc_loc[1].bias
print(w.shape)
#w.set_data(nd.zeros(w.shape))
b.set_data(nd.array([1, 0, 0, 0, 1, 0]))


net.hybridize()
print('net_hybridize:',net)


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

#data = mx.nd.random.uniform(shape=(4,3, 28,28)).as_in_context(ctx)   
#out = net(data)
#print(out)
##
#print('w:',w.data())
#print('b:',b.data())

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})
def train(epoch):
#   print(epoch)
   train_loss = 0.

   for batch_idx,(data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
#        print(data)
        label = label.as_in_context(ctx)
        batch_size = data.shape[0]			
#        print(batch_idx,batch_size)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data.dataset),
                100. * batch_idx / len(train_data), train_loss/len(train_data)))
   test()

#
## A simple test procedure to measure STN the performances on MNIST.
##
#
#
            
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()

def test():
    test_loss = 0.
    correct = 0.
    for data, label in test_data:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)

        output = net(data)
        loss = softmax_cross_entropy(output, label)
        # sum up batch loss
        test_loss += nd.mean(loss).asscalar()
        correct += accuracy(output, label)


    test_loss /= len(test_data.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct*batch_size, len(test_data.dataset),
                  100. * correct / len(test_data)))   

def visualize_stn():
    
#    batch,_ = test_data.dataset[:batch_size]
#    data = (batch.transpose(batch.astype('float32'), (0,3,1,2))/255).as_in_context(ctx)
    for i,(data,_) in enumerate(test_data):
        if i==1:
            break
        data = data.as_in_context(ctx)
        output = net.model[0](data)
        
        print(output.shape)
        print(output)
        
        in_grid = convert_image_np(make_grid(data))
        
        out_grid = convert_image_np(make_grid(output))
        
        # Plot the results side-by-side
        fig, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')
    
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')
     
        fig.savefig('result/compare.jpg',dpi=256,bbox_inches='tight', pad_inches = 0)
    
    
for epoch in range(1, 2):
    train(epoch)
#    test()            
#block = net                
#mx.viz.plot_network(block(mx.sym.var('img'))).view()

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()    