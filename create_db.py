# -*- coding: utf-8 -*-
"""
Created on Thu March 1 02:12:04 2018

@author: lav solanki
"""


import numpy as np
import cv2
import csv
import scipy.io
import argparse
from tqdm import tqdm
from utils import get_meta


def get_args():
    parser = argparse.ArgumentParser(description="cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--db", type=str, default="wiki")
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--min_score", type=float, default=4.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    db = args.db
    img_size = args.img_size
    min_score = args.min_score

    root_path = "data/wiki_crop/".format(db)
    mat_path = root_path + "wiki.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

    out_genders = []
    out_ages = []
    out_imgs = []
    out_faceScore=[]

    for i in tqdm(range(len(face_score))):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        out_genders.append(int(gender[i]))
        out_faceScore.append(int(face_score[i]))
        out_ages.append(age[i])
        img = cv2.imread(root_path + str(full_path[i][0]))
        out_imgs.append(cv2.resize(img, (img_size, img_size)))

    output = {"image": np.array(out_imgs), "gender": np.array(out_genders), "age": np.array(out_ages),
              "db": db, "img_size": img_size, "fscore": np.array(out_faceScore)}
    scipy.io.savemat(output_path, output)
    
   
   
    with open('mycsvfilet.csv', 'w') as f:  # Just use 'w' mode in 3.x
       w = csv.DictWriter(f, output.keys())
       w.writeheader()
       w.writerow(output)


if __name__ == '__main__':
    main()
