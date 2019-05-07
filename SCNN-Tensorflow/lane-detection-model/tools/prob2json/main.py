import cv2
import numpy as np
import os
import json
import pathlib

thr = 0.3
# prob_width, prob_height = 608, 384
# img_width, img_height = 1920, 1200
prob_width, prob_height = 800, 288
img_width, img_height = 1640, 590
interval = 16
exp = 'vgg_SCNN_DULR_w9'
pts = int(img_height / interval)
data = '/home/yujincheng/SCNN-Tensorflow/SCNN-Tensorflow/lane-detection-model/CULane'
prob_root = os.path.join('/home/yujincheng/SCNN-Tensorflow/SCNN-Tensorflow/lane-detection-model/predicts', exp, 'CULane')
img_path = '/home/yujincheng/SCNN-Tensorflow/SCNN-Tensorflow/lane-detection-model/CULane'
output = os.path.join('./output/', exp)
test_list = os.path.join(data, 'test.txt')

out = True
show = False
wait = 0
colors = [[0, 255, 0], [255, 0, 0], [0, 0, 255], [0, 255, 255]]
lane_type = ['l2', 'l1', 'r1', 'r2']


def get_lane(score):
    coordinate = np.zeros((1, pts)) - 2
    for i in range(pts):
        line_id = np.uint16(prob_height - i * interval * prob_height / img_height - 1)
        line = score[line_id, :]
        idx = np.argmax(line)
        if line[idx] / 255 > thr:
            coordinate[:, i] = idx * img_width/prob_width
    if np.sum(coordinate > -1) < 2:
        coordinate = np.zeros((1, pts)) - 2
    return coordinate


if __name__ == '__main__':
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
    with open(test_list, 'r') as file:
        img_names = file.read().splitlines()
    json_result = []
    if show:
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('test', int(img_width/2), int(img_height/2))
    for img_name in img_names:
        if out:
            true_name = img_name.split('/')
            del true_name[1]
            true_name[0], true_name[2] = true_name[2], 'camera'
            img_json = {'image_name':os.path.join(*true_name)}
        coordinates = np.zeros((4, pts))
        with open(os.path.join(prob_root, img_name[1:-3] + 'exist.txt')) as exist:
            exists = exist.read().split(' ')[:-1]
        for n, exist in enumerate(exists):
            if exist == '1':
                score_path = os.path.join(prob_root, img_name[1:-4] + '_' + str(n + 1) + '_avg.png')
                score_map = cv2.imread(score_path, 0)
                coordinates[n, :] = get_lane(score_map)
        if show:
            img = cv2.imread(os.path.join(img_path, img_name[1:]))
            for n, exist in enumerate(exists):
                if exist == '1':
                    for p in range(pts):
                        if coordinates[n, p] > 0:
                            cv2.circle(img, (int(coordinates[n, p]),
                                             int(img_height - p * interval)), 10, colors[n], -1)
            cv2.imshow('test', img)
            cv2.waitKey(wait)
        if out:
            lanes_json = []
            lanes_txt = []
            for n, exist in enumerate(exists):
                lane_json = {'xs': np.flip(coordinates[n, :]).astype(int).tolist(),
                            'key_lane_type': lane_type[n]}
                lanes_json.append(lane_json)
                if exist == '1' and np.sum(coordinates[n, :] > -1) > 1:
                    lanes_txt.append('')
                    for p in range(pts):
                        if coordinates[n, p] > 0:
                            lanes_txt[-1] += '{:d} {:d} '.format(int(coordinates[n, p]), int(img_height - p * interval))
            img_json['lanes'] = lanes_json
            img_json['h_samples'] = [h * interval for h in range(pts)]
            json_result.append(img_json)
            txt_name = os.path.join(output, img_name[1:-4] + '.lines.txt')
            pathlib.Path(os.path.dirname(txt_name)).mkdir(parents=True, exist_ok=True)
            with open(txt_name, 'w') as file:
                file.write('\n'.join(lanes_txt))

        print(img_name)

    if out:
        with open(os.path.join(output, 'result.json'), 'w') as file:
            file.write(json.dumps(json_result))
