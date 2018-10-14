import argparse
import logging
import time
import sys

import cv2
import numpy as np

from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path, model_wh

from utils.sort import Sort
from utils.actions import actionPredictor
from utils.joint_preprocess import *

import settings

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
mode = {'Pose Estimation': 'estimation',
        'Tracking': 'tracking',
        'Action Recognition': 'recognition'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--mode', type=str, default='estimation', help='Multi-person pose estimation')

    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    #logger.debug('cam read+')

    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()

    #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    tracker = Sort(settings.sort_max_age, settings.sort_min_hit)

    # Object label container for action recognition
    current = []
    previous = []
    memory = {}
    data = {}

    while True:
        ret_val, image = cam.read()

        image = cv2.resize(image, (settings.winWidth, settings.winHeight))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if ret_val and args.mode==mode['Pose Estimation']:
            #logger.debug('image process+')
            humans = estimator.inference(image) 

            #logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)   

        elif ret_val and args.mode==mode['Tracking']:
            humans = estimator.inference(image)
            image, joints, bboxes, xcenter, sk= TfPoseEstimator.get_skeleton(image, humans, imgcopy=False)
            height = image.shape[0]
            width = image.shape[1]

            if bboxes:
                result = np.array(bboxes)
                det = result[:, 0:5]
                det[:, 0] = det[:, 0] * width
                det[:, 1] = det[:, 1] * height
                det[:, 2] = det[:, 2] * width
                det[:, 3] = det[:, 3] * height
                trackers = tracker.update(det)

                for d in trackers:
                    xmin = int(d[0])
                    ymin = int(d[1])
                    xmax = int(d[2])
                    ymax = int(d[3])
                    label = int(d[4])
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                                    (int(settings.c[label % 32, 0]),
                                    int(settings.c[label % 32, 1]),
                                    int(settings.c[label % 32, 2])), 4) 

        elif ret_val and args.mode==mode['Action Recognition']:
            humans = estimator.inference(image)
            img_ori = np.copy(image)

            image, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(image, humans, imgcopy=False)
            height = image.shape[0]
            width = image.shape[1]

            if bboxes:
                result = np.array(bboxes)
                det = result[:, 0:5]
                det[:, 0] = det[:, 0] * width
                det[:, 1] = det[:, 1] * height
                det[:, 2] = det[:, 2] * width
                det[:, 3] = det[:, 3] * height
                trackers = tracker.update(det)
                current = [i[-1] for i in trackers]

                if len(previous) > 0:
                    for item in previous:
                        if item not in current and item in data:
                            del data[item]
                        if item not in current and item in memory:
                            del memory[item]

                previous = current

                for d in trackers:
                    xmin = int(d[0])
                    ymin = int(d[1])
                    xmax = int(d[2])
                    ymax = int(d[3])
                    label = int(d[4])
                    try:
                        j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
                    except:
                        j = 0

                    if joint_filter(joints[j]):
                        joints[j] = joint_completion(joint_completion(joints[j]))
                        if label not in data:
                            data[label] = [joints[j]]
                            memory[label] = 0
                        else:
                            data[label].append(joints[j])

                        if len(data[label]) == settings.L:
                            pred = actionPredictor().move_status(data[label])
                            if pred == 0:
                                pred = memory[label]
                            else:
                                memory[label] = pred
                            data[label].pop(0)

                            location = data[label][-1][1]
                            if location[0] <= 30:
                                location = (51, location[1])
                            if location[1] <= 10:
                                location = (location[0], 31)

                            cv2.putText(image, settings.move_status[pred], (location[0] - 30, location[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)

                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                    	         (int(settings.c[label % 32, 0]),
                    	         	int(settings.c[label % 32, 1]),
                    	         	int(settings.c[label % 32, 2])), 4)

        else:
            sys.exit('Abort...please choose correct running mode (estimation / tracking / recognition')

        cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time() 

        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')

    cv2.destroyAllWindows()