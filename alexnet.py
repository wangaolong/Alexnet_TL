# coding: UTF-8



import tensorflow as tf
import numpy as np
# define different layer functions
# we usually don't do convolution and pooling on batch and channel
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
    """dropout"""
    return tf.nn.dropout(x, keepPro, name)

def LRN(x, R, alpha, beta, name = None, bias = 1.0):
    """LRN"""
    return tf.nn.local_response_normalization(x, depth_radius = R, alpha = alpha,
                                              beta = beta, bias = bias, name = name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = "SAME", groups = 1):
    """convlutional"""
    channel = int(x.get_shape()[-1]) #x的第一个维度是1,是因为这里不是在训练,是在测试,所以一张一张放进去
    conv = lambda a, b: tf.nn.conv2d(a, b, strides = [1, strideY, strideX, 1], padding = padding)
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel/groups, featureNum])
        b = tf.get_variable("b", shape = [featureNum])

        #分组切成groups份
        xNew = tf.split(value = x, num_or_size_splits = groups, axis = 3)
        wNew = tf.split(value = w, num_or_size_splits = groups, axis = 3)

        #使用zip()函数将xNew和wNew中对应的元素打包成一个个元组
        #将对应的x和w放到设置的lambda表达式中进行卷积
        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        #将group切分开的合并在一起
        mergeFeatureMap = tf.concat(axis = 3, values = featureMap)
        #将合并在一起的x,w卷积添加上偏置b
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out, mergeFeatureMap.get_shape().as_list()), name = scope.name)

class alexNet(object):
    """alexNet model"""
    def __init__(self, x, keepPro, classNum, skip, modelPath = "bvlc_alexnet.npy"):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath #加载进来Alexnet的相关参数文件路径
        #build CNN
        self.buildCNN() #通过这个函数来创建CNN模型

    #创建CNN模型
    def buildCNN(self):
        """build model"""
        '''
        11和11是卷积核大小,可以参照1.png中的模型来看窗口大小是11*11
        4和4是卷积核步长,原图上写了Stride of 4
        96是要将color_channel3映射成多少维,原图是48,但是GPU速度限制的问题,分成两个48并行运算了
        '''
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        pool1 = maxPoolLayer(conv1, 3, 3, 2, 2, "pool1", "VALID")
        #局部响应归一化
        lrn1 = LRN(pool1, 2, 2e-05, 0.75, "norm1")

        conv2 = convLayer(lrn1, 5, 5, 1, 1, 256, "conv2", groups = 2)
        pool2 = maxPoolLayer(conv2, 3, 3, 2, 2, "pool2", "VALID")
        lrn2 = LRN(pool2, 2, 2e-05, 0.75, "lrn2")

        conv3 = convLayer(lrn2, 3, 3, 1, 1, 384, "conv3")

        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups = 2)

        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups = 2)
        pool5 = maxPoolLayer(conv5, 3, 3, 2, 2, "pool5", "VALID")

        #全连接层
        fcIn = tf.reshape(pool5, [-1, 256 * 6 * 6])
        fc1 = fcLayer(fcIn, 256 * 6 * 6, 4096, True, "fc6")
        dropout1 = dropout(fc1, self.KEEPPRO)

        fc2 = fcLayer(dropout1, 4096, 4096, True, "fc7")
        dropout2 = dropout(fc2, self.KEEPPRO)

        self.fc3 = fcLayer(dropout2, 4096, self.CLASSNUM, True, "fc8")

    def loadModel(self, sess):
        """load model"""
        wDict = np.load(self.MODELPATH, encoding = "bytes").item()
        #for layers in model
        for name in wDict:
            if name not in self.SKIP: #如果没有声明跳过此层
                with tf.variable_scope(name, reuse = True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            #bias
                            #不可训练OP
                            #将p赋值为偏置b
                            #shape: (4096, )
                            sess.run(tf.get_variable('b', trainable = False).assign(p))
                        else:
                            #weights
                            #shape: (9216, 4096)
                            sess.run(tf.get_variable('w', trainable = False).assign(p))
