# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:56:55 2018

@author: feywell
"""

import mxnet as mx
from mxnet import sym
import logging
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

mnist = mx.test_utils.get_mnist()

batch_size = 64
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

def get_loc(data, attr={'lr_mult':'0.01'}):
    """
    the localisation network in lenet-stn, it will increase acc about more than 1%,
    when num-epoch >=15
    """
    loc = sym.Convolution(data=data, num_filter=8, kernel=(7, 7), stride=(1,1))
    loc = sym.Activation(data = loc, act_type='relu')
    loc = sym.Pooling(data=loc, kernel=(2, 2), stride=(2, 2), pool_type='max')
    loc = sym.Convolution(data=loc, num_filter=10, kernel=(5, 5), stride=(1,1))
    loc = sym.Activation(data = loc, act_type='relu')
    loc = sym.Pooling(data=loc, kernel=(2, 2),stride=(2, 2), pool_type='max')
    
    loc = sym.FullyConnected(data=loc, num_hidden=32, name="stn_loc_fc1", attr=attr)
    loc = sym.Activation(data = loc, act_type='relu')
#       loc = sym.Flatten(data=loc)
    loc = sym.FullyConnected(data=loc, num_hidden=6, name="stn_loc_fc2", attr=attr)
    return loc
    
def get_symbol(num_classes=10, flag='training' ,add_stn=False, **kwargs):
    data = sym.Variable('data')
    if add_stn:
        data = sym.SpatialTransformer(data=data, loc=get_loc(data), target_shape = (28,28),
                                         transform_type="affine", sampler_type="bilinear")
    # first conv
    conv1 = sym.Convolution(data=data, kernel=(5,5), num_filter=10)
    relu1 = sym.Activation(data=conv1, act_type="relu")
    pool1 = sym.Pooling(data=relu1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = sym.Convolution(data=pool1, kernel=(5,5), num_filter=20)
    relu2 = sym.Activation(data=conv2, act_type="relu")
    pool2 = sym.Pooling(data=relu2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    
    drop1 = mx.sym.Dropout(data=pool2)
    # first fullc
    flatten = sym.Flatten(data=drop1)
    fc1 = sym.FullyConnected(data=flatten, num_hidden=50)
    relu3 = sym.Activation(data=fc1, act_type="relu")
    # second fullc
    drop2 = mx.sym.Dropout(data=relu3,mode=flag)
    fc2 = sym.FullyConnected(data=drop2, num_hidden=num_classes)
    # loss
    lenet = sym.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

stn_net = mx.mod.Module(symbol = get_symbol(),context=mx.gpu())

stn_net.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=20       
        )

test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
prob = stn_net.predict(test_iter)
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
# predict accuracy for lenet
acc = mx.metric.Accuracy()
stn_net.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.98, "Achieved accuracy (%f) is lower than expected (0.98)" % acc.get()[1]