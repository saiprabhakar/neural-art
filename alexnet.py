import tensorflow as tf
import numpy as np
from skimage import io, transform

# Build alexnet from weight and model dictionaries
# ('conv', fsize, fno, stride, padding, group)
# ('lrn', depth_radius, alpha, beta, bias)
# ('maxpool', ksize, stride, padding)
# ('fc', nunits)


def load_process_image(imgpath, mean_pix):
    # Load, resize, meansub and rgb->bgr transformation
    # mean_pix in bgr format
    # returns orgimage and processed img
    im1 = []
    im1 = io.imread(imgpath)[:,:,:3].astype(np.double)
    im1 = transform.resize(im1, (227,227,3)).astype(np.float32)
    im0 = np.copy(im1)
    temp1 = np.copy(im1[:, :, 2])
    temp2 = np.copy(im1[:, :, 0])
    im1[:, :, 0] = temp1
    im1[:, :, 2] = temp2
    im1 = im1 - mean_pix#np.mean(im1)
    return im0, im1

def unprocess_image(im1, mean_pix):
    im1 += mean_pix
    temp1 = np.copy(im1[:, :, 2])
    temp2 = np.copy(im1[:, :, 0])
    im1[:, :, 0] = temp1
    im1[:, :, 2] = temp2
    return im1

def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]), pred[0])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return pred[0]

def _conv(_input, name, w, b, stride, padding, group):
    '''
    Build conv layer
    '''
    fsize = w.shape[0]
    fno = w.shape[3]
    with tf.name_scope(name):
        cW = tf.Variable(w)
        cb = tf.Variable(b)
        if group == 1:
            conv = tf.nn.conv2d(_input, cW, [1, stride, stride, 1],\
                      padding=padding)
        elif group == 2:
            input_groups =  tf.split(_input, group, 3) 
            w_groups = tf.split(w, group, 3)
            output_groups = [tf.nn.conv2d(i, k, [1, stride, stride, 1],\
                      padding=padding) for i,k in zip(input_groups, w_groups)]
            conv = tf.concat(output_groups, 3)        
        else:
            raise InputError("wrong group")
        conv = tf.nn.bias_add(conv, cb)
        conv = tf.reshape(conv, [-1]+conv.get_shape().as_list()[1:])
        conv_relu = tf.nn.relu(conv)
    return conv_relu

def _lrn(_input, name, radius, alpha, beta, bias):
    with tf.name_scope(name):
        return tf.nn.local_response_normalization(_input,\
                                                 depth_radius= radius,
                                                 alpha=alpha,
                                                 beta=beta,
                                                 bias=bias)

def _maxpool(_input, name, ksize, stride, padding):
    with tf.name_scope(name):
        return tf.nn.max_pool(_input, ksize=[1, ksize, ksize, 1], 
                      strides=[1, stride, stride, 1],
                      padding=padding)

def _fc(_input, name, nunits, acttype, w, b):
    with tf.name_scope(name):
        if len(_input.shape) > 2:
            _input = tf.reshape(_input, 
                    [-1, int(np.prod(_input.get_shape()[1:]))])
        cW = tf.Variable(w)
        cb = tf.Variable(b)
        if acttype == "relu":
            return tf.nn.relu_layer(_input, cW, cb)
        elif acttype == "lin":
            return tf.nn.xw_plus_b(_input, cW, cb)
        else:
            raise InputError('notsupported activation type')

def build_alexnet(x, net_data):
    net_config = {  'conv1': ["conv", 11, 96, 4, 'SAME', 1],
                    'lrn1': ["lrn", 2, 2e-05, 0.75, 1.0],
                    'maxpool1': ["maxpool", 3, 2, 'VALID'],
                    'conv2': ["conv", 5, 256, 1, 'SAME', 2],
                    'lrn2': ["lrn", 2, 2e-05, 0.75, 1.0],
                    'maxpool2': ["maxpool", 3, 2, 'VALID'],
                    'conv3': ["conv", 3, 384, 1, 'SAME', 1],
                    'conv4': ["conv", 3, 384, 1, 'SAME', 2],
                    'conv5': ["conv", 3, 256, 1, 'SAME', 2],
                    'maxpool5': ["maxpool", 3, 2, 'VALID'],
                    'fc6': ["fc", 'relu', 4096],
                    'fc7': ["fc", 'relu', 4096],
                    'fc8': ["fc", 'lin', 1000]}
    layer_names = ['conv1', 'lrn1', 'maxpool1',
                  'conv2', 'lrn2', 'maxpool2',
                  'conv3', 'conv4', 'conv5', 'maxpool5',
                  'fc6', 'fc7', 'fc8']
    current = x

    for lname in layer_names:
        if net_config[lname][0] == 'conv':
            current = _conv(current, lname, net_data[lname][0], net_data[lname][1],
                  net_config[lname][3], net_config[lname][4], net_config[lname][5])
        elif net_config[lname][0] == 'lrn': 
            current = _lrn(current, lname, net_config[lname][1], net_config[lname][2],
                          net_config[lname][3], net_config[lname][4])
        elif net_config[lname][0] == 'maxpool':
            current = _maxpool(current, lname, net_config[lname][1], net_config[lname][2], 
                               net_config[lname][3])
        elif net_config[lname][0] == 'fc':
            current = _fc(current, lname, net_config[lname][2], net_config[lname][1],
                         net_data[lname][0], net_data[lname][1])
        else:
            raise ValueError("unsupported layer type")
    prob = tf.nn.softmax(current)

    return prob