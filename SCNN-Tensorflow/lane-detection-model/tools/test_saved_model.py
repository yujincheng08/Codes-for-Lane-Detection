from tensorflow.python.platform import gfile

import os
import os.path as ops
import argparse
import math
import tensorflow as tf
import glog as log
import cv2

try:
    from cv2 import cv2
except ImportError:
    pass

from config import global_config
from data_provider import lanenet_data_processor_test

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--model_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=8)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()


def test_lanenet(image_path, model_path, use_gpu, image_list, batch_size, save_dir):
    """
    :param batch_size:
    :param image_list:
    :param image_path:
    :param model_path:
    :param use_gpu:
    :param save_dir:
    :return:
    """
    test_dataset = lanenet_data_processor_test.DataSet(image_path, batch_size)
    input_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensor')
    imgs = tf.map_fn(test_dataset.process_img, input_tensor, dtype=tf.float32)

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        model_input = sess.graph.get_tensor_by_name('img_input:0')
        binary_seg = sess.graph.get_tensor_by_name('binary_seg:0')
        instance_seg = sess.graph.get_tensor_by_name('instance_seg:0')
        sess.run(tf.global_variables_initializer())

        for i in range(math.ceil(len(image_list) / batch_size)):
            print(i)
            paths = test_dataset.next_batch()
            imgs_out = sess.run(imgs, feed_dict={input_tensor: paths})
            instance_seg_image, existence_output = sess.run([binary_seg, instance_seg],
                                                            feed_dict={model_input: imgs_out})
            for cnt, image_name in enumerate(paths):
                print(image_name)
                parent_path = os.path.dirname(image_name)
                directory = os.path.join(save_dir, 'vgg_SCNN_DULR_w9', parent_path)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_exist = open(os.path.join(directory, os.path.basename(image_name)[:-3] + 'exist.txt'), 'w')
                for cnt_img in range(4):
                    cv2.imwrite(os.path.join(directory,
                                             os.path.basename(image_name)[:-4] + '_' + str(cnt_img + 1) + '_avg.png'),
                                (instance_seg_image[cnt, :, :, cnt_img + 1] * 255).astype(int))
                    if existence_output[cnt, cnt_img] > 0.5:
                        file_exist.write('1 ')
                    else:
                        file_exist.write('0 ')

                file_exist.close()

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    save_dir = os.path.join(args.image_path, 'predicts')
    if args.save_dir is not None:
        save_dir = args.save_dir

    img_name = []
    with open(str(args.image_path), 'r') as g:
        for line in g.readlines():
            img_name.append(line.strip())

    test_lanenet(args.image_path, args.model_path, args.use_gpu, img_name, args.batch_size, save_dir)
