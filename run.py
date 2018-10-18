
import argparse
import cv2
from actions import actions

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