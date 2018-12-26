from lanenet_model import lanenet_merge_model

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib
import argparse
import os

from config import global_config

CFG = global_config.cfg


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)

    return parser.parse_args()


def save_model(args):
    sess_config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(graph=tf.Graph(), config=sess_config) as sess:

        input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3],
                                      name='img_input')

        phase = tf.constant('test', tf.string)

        net = lanenet_merge_model.LaneNet()

        binary_seg_ret, instance_seg_ret = net.test_inference(input_tensor, phase, 'lanenet_loss')

        sess.run(tf.global_variables_initializer())
        initial_var = tf.global_variables()
        final_var = initial_var[:-1]
        saver = tf.train.Saver(final_var)

        saver.restore(sess=sess, save_path=args.weights_path)

        tf.identity(binary_seg_ret, 'binary_seg')
        tf.identity(instance_seg_ret, 'instance_seg')
        # fix batch norm nodes
        gd = sess.graph.as_graph_def()
        for node in gd.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        constant_graph = graph_util.convert_variables_to_constants(sess, gd,
                                                                   ['binary_seg', 'instance_seg'])

        os.makedirs(args.save_dir, exist_ok=True)
        with tf.gfile.FastGFile(os.path.join(args.save_dir, 'model.pb'), mode='wb') as f:
            f.write(constant_graph.SerializeToString())


if __name__ == '__main__':
    save_model(init_args())
