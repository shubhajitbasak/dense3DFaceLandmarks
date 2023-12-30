import time
import cv2
import os
import glob
import argparse
import numpy as np
import torch
import timm


from utils.utils_inference import plot_kpt, plot_vertices
from utils.utils_inference import process_input_mp
from utils.util import get_config

def main(config_file):
    cfg = get_config(config_file)
    local_rank = args.local_rank
    isGPU = True

    # ---- load detectors
    if cfg.is_dlib:
        import dlib
        detector_path = 'data/net-data/mmod_human_face_detector.dat'
        face_detector = dlib.cnn_face_detection_model_v1(
            detector_path)

    if cfg.is_mp:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
    checkpoint_path = '/mnt/satan/code/myGit/3DFaceLight/MyTest/checkpoints/mobile_net/Oct21/net_39.pth'

    if isGPU:
        net = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=cfg.num_verts * 3).to(local_rank)

    else:
        net = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=cfg.num_verts * 3)

    net.load_state_dict(torch.load(checkpoint_path))
    net.eval()

    # inference
    if not cfg.use_cam:  # on test-img
        print("[*] Processing on images in {}. Press 's' to save result.".format(cfg.eval_img_path))
        img_paths = glob.glob(os.path.join(cfg.eval_img_path, '*.jpg'))
        for img_path in img_paths:
            print(img_path)

            img = cv2.imread(img_path)
            vertices = process_input_mp(img_path, net, mp_face_detection, cuda=isGPU)
            if vertices is None:
                continue

            kpt_filter = np.loadtxt(cfg.filtered_kpt_500).astype(int)

            kpt = vertices[kpt_filter, :]

            result_list = [img,
                           plot_vertices(img, vertices),
                           plot_kpt(img, kpt)]

            cv2.imshow('Input', result_list[0])
            cv2.imshow('Sparse alignment', result_list[1])
            cv2.imshow('Sparse alignment GT', result_list[2])
            cv2.moveWindow('Input', 0, 0)
            cv2.moveWindow('Sparse alignment', 500, 0)
            cv2.moveWindow('Sparse alignment GT', 1000, 0)
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit()
            elif key == ord('s'):
                save_path = os.path.dirname(checkpoint_path)
                cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)),
                            np.concatenate(result_list, axis=1))
                print("Result saved in {}".format(save_path))
    else:  # webcam demo
        cap = cv2.VideoCapture(0)

        start_time = time.time()
        count = 1
        while (True):
            _, image = cap.read()

            pos = process_input_mp(image, net, mp_face_detection, cuda=isGPU)
            fps_str = 'FPS: %.2f' % (1 / (time.time() - start_time))
            start_time = time.time()
            cv2.putText(image, fps_str, (25, 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

            key = cv2.waitKey(1)
            if pos is None:
                cv2.waitKey(1)
                cv2.destroyWindow('Sparse alignment')
                cv2.destroyWindow('Dense alignment')
                cv2.destroyWindow('Pose')
                if key & 0xFF == ord('q'):
                    break
                continue

            else:
                vertices = pos
                kpt_filter = np.loadtxt(cfg.filtered_kpt_500).astype(int)
                kpt = vertices[kpt_filter, :]
                result_list = [image,
                               plot_vertices(image, vertices),
                               plot_kpt(image, kpt)]

                cv2.imshow('Input', result_list[0])
                cv2.imshow('Sparse alignment', result_list[1])
                cv2.imshow('Sparse alignment GT', result_list[2])
                cv2.moveWindow('Input', 0, 0)
                cv2.moveWindow('Sparse alignment', 500, 0)
                cv2.moveWindow('Sparse alignment GT', 1000, 0)

                if key & 0xFF == ord('q'):
                    cap.release()
                    # out.release()
                    cv2.destroyAllWindows()
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--configfile', default='configs/config.py', help='path to the configfile')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    print(os.getcwd())
    args = parser.parse_args()

    main(args.configfile)
