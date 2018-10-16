import argparse
import logging
import time
import sys

import cv2
import numpy as np

from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path, model_wh

from utils.sort import Sort
from utils.actions import actionPredictor, actionPredictor_params
from utils.joint_preprocess import *

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class actions(object):

    def __init__(self, arguments):
        self.arguments = arguments

        # Frame window dim
        self.winWidth =640
        self.winHeight = 480

        actionPredictor_params.__init__(self)

        self.fps_time = 0
        #self.step = 15
        self.mode = {'Pose Estimation': 'estimation',
                     'Tracking': 'tracking',
                     'Action Recognition': 'recognition'}

        w, h = model_wh(self.arguments.resize)
        if w > 0 and h > 0:
            self.estimator = TfPoseEstimator(get_graph_path(self.arguments.model), target_size=(w, h))
        else:
            self.estimator = TfPoseEstimator(get_graph_path(self.arguments.model), target_size=(432, 368))

        self.cam = cv2.VideoCapture(self.arguments.camera)

        # Tracker based on Sort
        self.sort_max_age = 20
        self.sort_min_hit = 3
        self.tracker = Sort(self.sort_max_age, self.sort_min_hit)

    def proceed(self):
        self._read_frame_()

        if self.ret_val and self.arguments.mode == self.mode['Pose Estimation']:
            self._perform_estimation_()

        elif self.ret_val and self.arguments.mode == self.mode['Tracking']:
            self._perform_tracking_()

        elif self.ret_val and self.arguments.mode == self.mode['Action Recognition']:
            self._perform_recognition_()

        else:
            sys.exit('Abort...please choose correct action mode from "estimation" "tracking" "recognition"')

        self._output_()


    def _perform_estimation_(self):
        self.humans = self.estimator.inference(self.image)
        self.image = TfPoseEstimator.draw_humans(self.image, self.humans, imgcopy=False)

    def _perform_tracking_(self):
        self.humans = self.estimator.inference(self.image)
        self.image, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(self.image, self.humans, imgcopy=False)

        height = self.image.shape[0]
        width = self.image.shape[1]

        if bboxes:
            result = np.array(bboxes)
            det = result[:, 0:5]
            det[:, 0] = det[:, 0] * width
            det[:, 1] = det[:, 1] * height
            det[:, 2] = det[:, 2] * width
            det[:, 3] = det[:, 3] * height
            trackers = self.tracker.update(det)

            for d in trackers:
                xmin = int(d[0])
                ymin = int(d[1])
                xmax = int(d[2])
                ymax = int(d[3])
                label = int(d[4])
                cv2.rectangle(self.image, (xmin, ymin), (xmax, ymax),
                                (int(self.c[label % 32, 0]),
                                int(self.c[label % 32, 1]),
                                int(self.c[label % 32, 2])), 4) 

    def _perform_recognition_(self):

        self.predictor = actionPredictor()

        self.humans = self.estimator.inference(self.image)
        self.image, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(self.image, self.humans, imgcopy=False)

        height = self.image.shape[0]
        width = self.image.shape[1]

        if bboxes:
            result = np.array(bboxes)
            det = result[:, 0:5]
            det[:, 0] = det[:, 0] * width
            det[:, 1] = det[:, 1] * height
            det[:, 2] = det[:, 2] * width
            det[:, 3] = det[:, 3] * height
            trackers = self.tracker.update(det)

            self.current = [i[-1] for i in trackers]

            if len(self.previous) > 0:
                for item in self.previous:
                    if item not in self.current and item in self.data:
                        del self.data[item]
                    if item not in self.current and item in self.memory:
                        del self.memory[item]
            self.previous = self.current

            for d in trackers:
                xmin = int(d[0])
                ymin = int(d[1])
                xmax = int(d[2])
                ymax = int(d[3])
                label = int(d[4])

                logger.debug('label is: %d' % (label))

                try:
                    j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
                except:
                    j = 0

                if joint_filter(joints[j]):
                    joints[j] = joint_completion(joint_completion(joints[j]))

                    if label not in self.data:
                        logger.debug('label is: %d' % (label))

                        self.data[label] = [joints[j]]
                        self.memory[label] = 0
                    else:
                        self.data[label].append(joints[j])

                    if len(self.data[label]) == self.step:
                        pred = self.predictor.move_status(self.data[label])

                        logger.debug(len(self.data[label]))

                        if pred == 0:
                            pred = self.memory[label]
                        else:
                            self.memory[label] = pred
                        self.data[label].pop(0)

                        location = self.data[label][-1][1]
                        #location = functools.reduce(lambda x, y: x + y, self.data[label][:][1]) / len(self.data[label][:][1])
                        #location = sum(self.data[label][:][1]) / float(len(self.data[label][:][1]))

                        if location[0] <= 30:
                            location = (51, location[1])
                        if location[1] <= 10:
                            location = (location[0], 31)

                        cv2.putText(self.image, self.move_status[pred], (location[0] - 30, location[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

                cv2.rectangle(self.image, (xmin, ymin), (xmax, ymax),
                             (int(self.c[label % 32, 0]),
                                int(self.c[label % 32, 1]),
                                int(self.c[label % 32, 2])), 4)

    def _read_frame_(self):
        self.ret_val, self.image = self.cam.read()

        self.image = cv2.resize(self.image, (self.winWidth, self.winHeight))

    def _output_(self):
        # Calculate frame averaging step
        FPS = float(1.0 / (time.time() - self.fps_time))
        logger.debug('FPS: %f' % FPS)


        #self.step = int(0.7 * FPS)
        #logger.debug('step: %d' % self.step)

        cv2.putText(self.image,
                        "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', self.image)
        self.fps_time = time.time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--mode', type=str, default='estimation', help='Multi-person pose estimation')

    args = parser.parse_args()

    act = actions(args)

    while True:
        # Perform user intended action
        act.proceed()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()