#!/usr/bin/env python
# coding: UTF-8
import os
import urllib.request
import argparse
import sys
import alexnet
import cv2
#https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
#UC浏览器有PYPI网站书签,3.4.1版本在那里下的
#谷歌浏览器下载这种有
import tensorflow as tf
import numpy as np
import caffe_classes
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

'''
参数解析器模块argparse使用方法
第一步:import argparse
第二步:使用argparse.ArgumentParser()函数 创建parser
第三步:使用parser.add_argument()函数 向parser中添加若干参数
第四步:使用parser.parse_args加载sys.argv参数列表
'''
'''
sys.argv:
这是系统参数,需要import sys模块
第一个参数是script path,本例中是D:\PyCharm_Project\Alexnet\testAlex.py
其实这些参数都是在Edit configurations中进行配置的

sys.argv列表后续的参数是使用add_argument添加进来的配置参数
'''
'''
argparse.ArgumentParser()函数:
这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确
(此时python解释器其实也是调用了pring_help()方法)时，会打印这些描述信息，
一般只需要传递description参数
'''
parser = argparse.ArgumentParser(description='Classify some images.')
'''
add_argument()方法:
help：和ArgumentParser方法中的参数作用相似，出现的场合也一致
default: 默认值
choices: 选择
nargs：命令行参数的个数

注意:
其中命令行参数如果没给定，且没有设置default，则出错。但是如果是选项的话，则设置为None
解释一下这句话:
①如果add_argument添加了一个参数,但是没有配置,如果配置了default,那么自动设置为default
②如果add_argument添加了一个参数,但是没有配置,也没有default,那么出错
③如果add_argument添加了一个参数,而且是选项choices的话,如果没有配置,则设置为None
'''
parser.add_argument('mode', choices=['folder', 'url'], default='folder')
parser.add_argument('path', help='Specify a path [e.g. testModel]')
args = parser.parse_args(sys.argv[1:])

#读取图像成
if args.mode == 'folder':
    #get testImage
    '''
    lambda表达式定义一个函数:
    传入f,返回images/f
    '''
    withPath = lambda f: '{}/{}'.format(args.path,f)
    '''
    python dict()函数: 构造字典
    dict()                        # 创建空字典
    {}
     
    dict(a='a', b='b', t='t')     # 传入关键字
    {'a': 'a', 'b': 'b', 't': 't'}
    
    dict(zip(['one', 'two', 'three'], [1, 2, 3]))   # 映射函数方式来构造字典
    {'three': 3, 'two': 2, 'one': 1} 
    
    dict([('one', 1), ('two', 2), ('three', 3)])    # 可迭代对象方式来构造字典
    {'three': 3, 'two': 2, 'one': 1}
    
    dict((a, b) for a in XXX if YYY)                # 循环方式来构造字典
    '''
    '''
    os.listdir()函数: 取出路径下所有文件名
    args.path是'images'
    os.listdir()函数是'images'这个路径下所有文件名,'llama.jpeg','sealion.jpeg','zebra.jpeg'
    '''
    '''
    os.path.isfile()函数: 
    判断相对路径是否是文件
    只要是文件就返回True
    只要是文件夹就返回False
    '''
    '''
    这行代码的整体逻辑:
    用文件名f 'llama.jpeg'和cv2读出来的图片np.array构造字典
    其中f是args.path 'images'文件夹下所有文件名
    每次用withPath连接成相对路径 'images/llama.jpeg',
    之后用isfile()判断一下是文件就添加字典
    '''
    testImg = dict((f, cv2.imread(withPath(f))) for f in os.listdir(args.path)
                   if os.path.isfile(withPath(f)))
elif args.mode == 'url':
    def url2img(url):
        '''url to image'''
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    testImg = {args.path:url2img(args.path)}

#dict.value() 以列表返回字典中的所有值
if testImg.values(): #只要有图片被加载进来
    #some params
    dropoutPro = 1 #dropout参数,但是由于这里是拿别人的网络做测试,所以不需要dropout
    classNum = 1000 #1000分类问题
    skip = [] #是否要去掉神经网络中的某些层1,但是由于这里是拿别人的网络做测试,所以不需要去掉任何层

    '''
    一会我们要做一个减图像均值的操作
    104,117,124分别是三张图片R,G,B通道的均值
    这个均值本来是要计算的,但是这里没有计算,直接拿过来用了
    
    注意,应该是resize成(227, 227)之后的均值,不要现在直接算均值
    '''
    imgMean = np.array([104, 117, 124], np.float)

    '''
    原始论文中这里是224,不过修改成227更好
    之所以第一个数字是1,而不是一个batch_size是因为我们现在在做测试
    因此要一个一个图片的放进去进行测试
    '''
    x = tf.placeholder("float", [1, 227, 227, 3])

    #网络结构需要制定一些参数
    model = alexnet.alexNet(x, dropoutPro, classNum, skip)
    #score的shape是(1, 1000)
    #代表1000种分类对应的神经元的输出值
    score = model.fc3 #最后一个全连接层输出
    softmax = tf.nn.softmax(score) #使用softmax函数分配概率
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)

        for key,img in testImg.items():
            '''
            #img preprocess
            这里resize成(227, 227)并不是说resize之后的shape就是(227, 227)
            原来的图片是(336, 500, 3),reshape中dtype参数设定为(227, 227)之后
            最终的结果应该是(227, 227, 3)
            因为这个方法是cv2中的专门对图像进行处理的函数
            并不会改变图片的color channel
            '''
            resized = cv2.resize(img.astype(np.float), (227, 227)) - imgMean
            maxx = np.argmax(sess.run(softmax, feed_dict = {x: resized.reshape((1, 227, 227, 3))}))
            res = caffe_classes.class_names[maxx]

            font = cv2.FONT_HERSHEY_SIMPLEX
            #为图片添加文本
            cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 255, 0), 2)
            print("{}: {}\n----".format(key,res))
            cv2.imshow("demo", img)
            cv2.waitKey(0)
