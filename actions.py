import argparse
import logging
import time
import sys

import cv2
import numpy as np

from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path, model_wh

from utils.sort import Sort

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class actionPredictor_params(object):
    def __init__(self):

        # Frame clip length
        self.step = 15

        #self.move_status = ['', 'stand', 'sit', 'walk', 'walk close', 'walk away', 'sit down', 'stand up']
        self.move_status = ['', 'Not Crossing', 'Not Crossing', 'Crossing', 'walk close', 'walk away', 'sit down', 'stand up']

        self.c = np.random.rand(32, 3) * 255

        # Object label container for action recognition
        self.current = []
        self.previous = []
        self.memory = {}
        self.data = {}

class actionPredictor(object):

    def __init__(self):
        pass

    #@staticmethod
    def move_status(self, joints):
        # All move and actions are based on the joints info difference from the newest frame and oldest frame

        init_x = float(joints[0][1][0] + joints[0][8][0] + joints[0][11][0]) / 3
        init_y = float(joints[0][1][1] + joints[0][8][1] + joints[0][11][1]) / 3
        end_x = float(joints[-1][1][0] + joints[-1][8][0] + joints[-1][11][0]) / 3
        end_y = float(joints[-1][1][1] + joints[-1][8][1] + joints[-1][11][1]) / 3

        # Upper body height change
        init_h1 = float(joints[0][8][1] + joints[0][11][1]) / 2 - joints[0][1][1]
        end_h1 = float(joints[-1][8][1] + joints[-1][11][1]) / 2 - joints[-1][1][1]
        # Upper body height change rate
        try:
            h1 = end_h1 / init_h1
        except:
            h1 = 0.0
        
        # Thigh height change
        init_h2 = (float(joints[0][9][1] + joints[0][12][1]) - float(joints[0][8][1] + joints[0][11][1])) / 2
        end_h2 = (float(joints[-1][9][1] + joints[-1][12][1]) - float(joints[-1][8][1] + joints[-1][11][1])) / 2
        # Thigh height change rate
        try:
            h2 = end_h2 / init_h2
        except:
            h2 = 0.0

        # Upper body center change
        xc = end_x - init_x
        yc = end_y - init_y

        if abs(xc) < 30. and abs(yc) < 20.:
            ty_1 = float(joints[-1][1][1])
            ty_8 = float(joints[-1][8][1] + joints[-1][11][1]) / 2
            ty_9 = float(joints[-1][9][1] + joints[-1][12][1]) / 2
            try:
                t = float(ty_8 - ty_1) / (ty_9 - ty_8)
            except:
                t = 0.0
            if h1 < 1.16 and h1 > 0.84 and h2 < 1.16 and h2 > 0.84:

                if t < 1.73:
                    return 1
                else:
                    return 2
            else:
                if t < 1.7:
                    if h1 >= 1.08:
                        return 4

                    elif h1 < 0.92:
                        return 5
                    else:
                        return 0
                else:
                    return 0
        elif abs(xc) < 30. and abs(yc) >= 30.:
            init_y1 = float(joints[0][1][1])
            init_y8 = float(joints[0][8][1] + joints[0][11][1]) / 2
            init_y9 = float(joints[0][9][1] + joints[0][12][1]) / 2

            end_y1 = float(joints[-1][1][1])
            end_y8 = float(joints[-1][8][1] + joints[-1][11][1]) / 2
            end_y9 = float(joints[-1][9][1] + joints[-1][12][1]) / 2
            try:
                init_yc = float(init_y8 - init_y1) / (init_y9 - init_y8)
            except:
                init_yc = 0.0
            try:
                end_yc = float(end_y8 - end_y1) / (end_y9 - end_y8)
            except:
                end_yc = 0.0
            th_yc = 0.1
            if yc >= 25 and abs(end_yc - init_yc) >= th_yc:
                return 6

            elif yc < -20 and abs(end_yc - init_yc) >= th_yc:
                return 7

            else:
                return 0

        elif abs(xc) > 30. and abs(yc) < 30.:
            return 3

        else:
            return 0

    def act_mv_avg(self, joints):
        pass

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

    def _joint_filter_(self, joint):
        if 1 not in joint:
            return False
        # Check exist of hip
        if 8 not in joint and 11 not in joint:
            return False
        # Check exist of knee
        if 9 not in joint and 12 not in joint:
            return False
        return True

    def _joint_complete_(self, joint):
        if 8 in joint and 11 not in joint:
            joint[11] = joint[8]
        elif 8 not in joint and 11 in joint:
            joint[8] = joint[11]
        if 9 in joint and 12 not in joint:
            joint[12] = joint[9]
        elif 9 not in joint and 12 in joint:
            joint[9] = joint[12]

        return joint

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

                #logger.debug('label is: %d' % (label))
                
                # Locate the current person object in current frame
                # Iterated thru xcenter for finding minimum distance between object center coord
                try:
                    j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
                except:
                    j = 0
                # Check if major skeleton points are existing
                if self._joint_filter_(joints[j]):
                    joints[j] = self._joint_complete_(self._joint_complete_(joints[j]))

                    if label not in self.data:
                        #logger.debug('label is: %d' % (label))

                        self.data[label] = [joints[j]]
                        self.memory[label] = 0
                    else:
                        self.data[label].append(joints[j])

                    if len(self.data[label]) == self.step:
                        pred = self.predictor.move_status(self.data[label])

                        #logger.debug(len(self.data[label]))

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
                            (255, 255, 255), 2)

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
        #logger.debug('FPS: %f' % FPS)

        #self.step = int(0.7 * FPS)
        #logger.debug('step: %d' % self.step)

        cv2.putText(self.image,
                        "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
        cv2.imshow('tf-pose-estimation result', self.image)
        self.fps_time = time.time()