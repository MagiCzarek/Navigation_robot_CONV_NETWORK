'''
This is a script for adding images to different folders to create keras model
'''

import argparse
import os
import cv2
import tensorflow as tf
import numpy as np

LEFT = ord('a')
RIGHT = ord('d')
FORWARD = ord('w')
BACKWARD = ord('s')
STOP = ord('q')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', default='C:\Obrazki')
    parser.add_argument('-o', '--output_folder', default='E:\pythonProject3\Data')
    return parser.parse_args()

def save_image(category, image, target_counts, target_paths):
    path = target_paths[category]
    count = str(target_counts[category]) + '.png'
    cv2.imwrite(os.path.join(path, count), image)
    target_counts[category] += 1


def main(args):
    global target_paths
    input_path = args.input_folder

    target_paths = {LEFT: os.path.join(args.output_folder, 'left'),
                    RIGHT: os.path.join(args.output_folder, 'right'),
                    FORWARD: os.path.join(args.output_folder, 'forward'),
                    BACKWARD: os.path.join(args.output_folder, 'backward')}

    target_counts = {LEFT: 0,
                     RIGHT: 0,
                     FORWARD: 0,
                     BACKWARD: 0}

    for filename in os.listdir(input_path):
        if filename.endswith('.png'):
            chosen = False
            img = cv2.imread(os.path.join(input_path, filename))
            resized_img = cv2.resize(img, (0, 0), fx=3, fy=3)
            cv2.imshow('Image', resized_img)

            while not chosen:
                choice = cv2.waitKey(0)
                if choice in target_paths:
                    save_image(choice, img, target_counts, target_paths)
                    noise = np.random.normal(0.0, 0.06, size=img.shape) * 255
                    noised_image = img + noise
                    save_image(choice, noised_image, target_counts, target_paths)
                    chosen = True
                elif choice == STOP:
                    break
            if choice == STOP:
                break





if __name__ == '__main__':
    main(parse_arguments())
