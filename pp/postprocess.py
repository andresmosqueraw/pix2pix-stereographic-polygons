import glob
import os
import cv2
import argparse
import shutil
import sys

import numpy as np
import csv


def merge(FLAGS):
  """

  """

  if not os.path.isdir(FLAGS.test_dir):
    raise ValueError("Could not find test_dir.")

  if not os.path.isdir(FLAGS.input_dir):
    raise ValueError("Could not find input_dir.")

  if not os.path.isdir(FLAGS.target_dir):
    raise ValueError("Could not find target_dir.")

  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  test_list = sorted(glob.glob(os.path.join(FLAGS.test_dir, '*.png')))
  input_list = sorted(glob.glob(os.path.join(FLAGS.input_dir, '*.png')))

  for inp in input_list:
    input_base = os.path.splitext(os.path.basename(inp))[0]
    input_word = input_base + "_"
    inp_img = cv2.imread(inp)

    inp_height, inp_width = inp_img.shape[:2]
    # inp_height, inp_width = 256*6, 256*11

    test_list_ex = [img for img in test_list if input_word in img]
    test_list_output = [img for img in test_list_ex if "-outputs" in img]

    img_row_max = 0
    img_col_max = 0

    # img_name_array1d = []
    img_array1d = []
    for img in test_list_output:
      img_base = os.path.splitext(os.path.basename(img))[0].replace('-outputs', '')

      img_row = int(img_base.rsplit("_", 2)[-2])-1
      img_col = int(img_base.rsplit("_", 2)[-1])-1

      if img_row > img_row_max:
        img_row_max = img_row
      if img_col > img_col_max:
        img_col_max = img_col
      # print(img_row_max, img_col_max)

      if (img_row == 0) & (img_col == 0):
        img_img = cv2.imread(img)
        # img_img = imread(img)
        img_height, img_width = img_img.shape[:2]

      # img_name_array1d.append(img)
      # img_name_array1d.append((img_row, img_col))
      img_array1d.append(cv2.imread(img))

    # img_name_array = [img_name_array1d[i:i+img_col_max+1] for i in range(0, len(img_name_array1d), img_col_max+1)]
    img_array = [img_array1d[i:i+img_col_max+1] for i in range(0, len(img_array1d), img_col_max+1)]

    # print(img_name_array)
    # print(len(img_name_array), len(img_name_array[0]))

    width = inp_width
    w_split = img_width
    height = inp_height
    h_split = img_height
    print(height, width)
    print(h_split, w_split)

    # if width <= w_split:
    if img_col_max == 0:
      pad_h = [(0, width)]
    else:
      n_h, rem_h = divmod(width, w_split)
      if rem_h == 0:
        overlap_h = 0
        err_h = 0
      else:
        overlap_h = (w_split - rem_h) // n_h
        sum_h = (w_split - overlap_h) * n_h + w_split
        err_h = sum_h - width
        n_h += 1
      pad_h = list(range(n_h))
      left_next = 0
      for i_h in range(n_h):
        pad_L = overlap_h // 2
        pad_R = overlap_h - pad_L

        left = left_next
        right = left + w_split

        if i_h == 0:
          left = 0
        elif i_h < err_h +1:
          left = pad_L + 1
        else:
          left = pad_L

        if i_h == n_h - 1:
          right = w_split
        else:
          right = w_split - pad_R
        # # left_next = right - overlap_h
        # elif i_h < err_h:
        #   left_next -= 1
        # print(i_h, left, right, left_next)
        pad_h[i_h] = (left, right)

      # print(pad_h)


    # if height <= h_split:
    if img_row_max == 0:
      pad_v = [(0, height)]
    else:
      n_v, rem_v = divmod(height, h_split)
      if rem_v == 0:
        overlap_v = 0
        err_v = 0
      else:
        overlap_v = (h_split - rem_v) // n_v
        sum_v = (h_split - overlap_v) * n_v + h_split
        err_v = sum_v - height
        n_v += 1
      pad_v = list(range(n_v))
      left_next = 0
      for i_v in range(n_v):
        pad_U = overlap_v // 2
        pad_B = overlap_v - pad_U

        left = left_next
        right = left + h_split

        if i_v == 0:
          left = 0
        elif i_v < err_v +1:
          left = pad_U + 1
        else:
          left = pad_U

        if i_v == n_v - 1:
          right = h_split
        else:
          right = h_split - pad_B

        pad_v[i_v] = (left, right)

      # print(pad_v)


    img_array_pad = [[0 for x in range(img_col_max+1)] for y in range(img_row_max+1)]
    # print(img_array_pad)
    for col in range(img_col_max+1):
      for row in range(img_row_max+1):
        img_array_pad[row][col] = img_array[row][col][pad_v[row][0]:pad_v[row][1], pad_h[col][0]:pad_h[col][1]]

    if img_row_max == 0:
      # img_array_h = [img_tmp for img_tmp_1 in img_array for img_tmp in img_tmp_1]
      img_merge = cv2.hconcat(img_array_pad[0])
    else:
      img_merge = cv2.vconcat([cv2.hconcat(img_array_h) for img_array_h in img_array_pad])

    print(img_merge.shape[:2])

    cv2.imwrite(os.path.join(FLAGS.output_dir, input_base + ".png"), img_merge)


def analyze(FLAGS):
  if not os.path.isdir(FLAGS.input_dir):
    raise ValueError("Could not find input_dir.")

  if not os.path.isdir(FLAGS.target_dir):
    raise ValueError("Could not find target_dir.")

  if not os.path.isdir(FLAGS.output_dir):
    raise ValueError("Could not find output_dir.")


  input_list = sorted(glob.glob(os.path.join(FLAGS.input_dir, '*.png')))

  target_list = sorted(glob.glob(os.path.join(FLAGS.target_dir, '*.png')))

  output_list = sorted(glob.glob(os.path.join(FLAGS.output_dir, '*.png')))

  # writecolumns = ["{}".format("file"), "TP", "FP", "FN", "TN", "Precision", "Recall", "IoU"]
  writecolumns0 = ["file", "Precision(px)", "Recall(px)", "IoU(px)"]
  with open(os.path.join(FLAGS.output_dir, FLAGS.output_data), 'w') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar='\\', lineterminator='\n')
    writer.writerow(writecolumns0)

  sum_TP = 0
  sum_FP = 0
  sum_FN = 0
  sum_TN = 0

  for inp, tar, out in zip(input_list, target_list, output_list):
    img_input = cv2.imread(inp)
    img_target = cv2.imread(tar)
    img_output = cv2.imread(out)

    # Verificar y ajustar las dimensiones de img_output para que coincidan con img_target
    if img_target.shape != img_output.shape:
        img_output = cv2.resize(img_output, (img_target.shape[1], img_target.shape[0]), interpolation=cv2.INTER_AREA)

    input_base = os.path.splitext(os.path.basename(inp))[0]

    print(inp, tar, out)

    img_overlay = img_target.copy()
    img_overlay2 = img_input.copy()
    img_contour = img_input.copy()

    cond_target_f = (img_target[:, :, 0] == 255) & \
                        (img_target[:, :, 1] == 255) & \
                        (img_target[:, :, 2] == 255)
    cond_target_p = np.logical_not(cond_target_f)

    cond_output_f = (img_output[:, :, 0] == 255) & \
                        (img_output[:, :, 1] == 255) & \
                        (img_output[:, :, 2] == 255)
    cond_output_p = np.logical_not(cond_output_f)

    cond_overlay_TP = np.logical_and(cond_output_p, cond_target_p)
    cond_overlay_FP = np.logical_and(cond_output_p, cond_target_f)
    cond_overlay_FN = np.logical_and(cond_output_f, cond_target_p)
    cond_overlay_TN = np.logical_and(cond_output_f, cond_target_f)

    img_overlay[cond_overlay_TP] = [0, 0, 255]
    img_overlay[cond_overlay_FP] = [0, 255, 0]
    img_overlay[cond_overlay_FN] = [255, 255, 0]
    img_overlay[cond_overlay_TN] = [255, 255, 255]

    overlay_name = os.path.join(FLAGS.output_dir, input_base + "_overlay.png")
    cv2.imwrite(overlay_name, img_overlay)


    for color in [(0, 0, 255), (0, 255, 0), (255, 255, 0)]:
      black = cv2.inRange(img_overlay, color, color)
      contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      img_contour = cv2.drawContours(img_contour, contours, -1, color, 1)

    img_overlay[cond_overlay_TN] = [0, 0, 0]
    img_overlay2 = cv2.addWeighted(img_contour, 1, img_overlay, 0.2, 0)
    overlay2_name = os.path.join(FLAGS.output_dir, input_base + "_overlay2.png")
    cv2.imwrite(overlay2_name, img_overlay2)

    px_TP = np.count_nonzero(cond_overlay_TP)
    px_FP = np.count_nonzero(cond_overlay_FP)
    px_FN = np.count_nonzero(cond_overlay_FN)
    px_TN = np.count_nonzero(cond_overlay_TN)

    precision_px = px_TP / (px_TP + px_FP)
    recall_px = px_TP / (px_TP + px_FN)
    iou_px = px_TP / (px_TP + px_FP + px_FN)

    # print(px_TP, px_FP, px_FN, px_TN, px_all, px, precision, recall, iou)
    writecolumns = [os.path.basename(overlay_name), "{:.4f}".format(precision_px), "{:.4f}".format(recall_px), "{:.4f}".format(iou_px)]
    with open(os.path.join(FLAGS.output_dir, FLAGS.output_data), 'a', encoding='cp932') as f:
      writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar='\\', lineterminator='\n')
      writer.writerow(writecolumns)

    sum_TP += px_TP
    sum_FP += px_FP
    sum_FN += px_FN
    sum_TN += px_TN

  precision_all = sum_TP / (sum_TP + sum_FP)
  recall_all = sum_TP / (sum_TP + sum_FN)
  iou_all = sum_TP / (sum_TP + sum_FP + sum_FN)

  writecolumns_all = ["sum",  "{:.4f}".format(precision_all), "{:.4f}".format(recall_all), "{:.4f}".format(iou_all)]
  with open(os.path.join(FLAGS.output_dir, FLAGS.output_data), 'a',encoding='cp932') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar='\\', lineterminator='\n')
    writer.writerow(writecolumns_all)


def main():

  parser = argparse.ArgumentParser()

  parser.add_argument('--test_dir', type=str, default='results/CS_pix2pix/test_latest/images_tf')
  parser.add_argument('--input_dir', type=str, default='org_image/input')
  parser.add_argument('--target_dir', type=str, default='org_image/target')
  parser.add_argument('--output_dir', type=str, default='output')
  parser.add_argument('--output_data', type=str, default='out.csv')
  
  parser.add_argument('--ignore_empty_px', action='store_true')
  parser.add_argument('--split_record', type=str, default='')
  parser.add_argument('--lower_th', type=float, default=0.0)
  parser.add_argument('--upper_th', type=float, default=1.0)
  parser.add_argument('--merge', action='store_true')
  parser.add_argument('--result', action='store_true')
  parser.add_argument('--debug', action='store_true')


  FLAGS = parser.parse_args()
  # print(FLAGS)

  merge(FLAGS)
  analyze(FLAGS)


if __name__ == '__main__':

  main()
