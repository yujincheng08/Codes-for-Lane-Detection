from lanenet_model import lanenet_merge_model

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils

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

        img_input = tf.placeholder(dtype=tf.uint8, shape=[None, 1200, 1920, 3],
                                   name='img_input')

        img_float = tf.cast(img_input, tf.float32)
        resized_image = tf.image.resize_bicubic(img_input, (CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH))

        std_image = tf.map_fn(lambda img: tf.subtract(img, [103.939, 116.779, 123.68]), resized_image)

        input_tensor = tf.map_fn(lambda img: img[..., ::-1], std_image)

        phase = tf.constant('test', tf.string)

        net = lanenet_merge_model.LaneNet()

        binary_seg_ret, instance_seg_ret = net.test_inference(input_tensor, phase, 'lanenet_loss')

        sess.run(tf.global_variables_initializer())
        initial_var = tf.global_variables()
        final_var = initial_var[:-1]
        saver = tf.train.Saver(final_var)

        saver.restore(sess=sess, save_path=args.weights_path)

        instance_seg = tf.greater(instance_seg_ret, 0.5)
        instance_seg_info = utils.build_tensor_info(instance_seg)
        outputs = {'instance_seg': instance_seg_info}

        for i in range(5):
            outputs['binary_seg_%d' % i] = utils.build_tensor_info(binary_seg_ret[:, :, :, i])

        img_input_info = utils.build_tensor_info(img_input)
        output_signature = signature_def_utils.build_signature_def(
            inputs={'img_input': img_input_info},
            outputs=outputs,
            method_name=signature_constants.PREDICT_METHOD_NAME)

        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(args.save_dir, 'savemodel'))
        builder.add_meta_graph_and_variables(sess, ['serve'], signature_def_map={'detect': output_signature})

    builder.save()


if __name__ == '__main__':
    save_model(init_args())
