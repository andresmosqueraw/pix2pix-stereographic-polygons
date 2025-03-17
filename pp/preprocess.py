import argparse
import glob
import os
import json

import cv2
import numpy as np


def preparation(src_dir, subset_name):
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Could not find {src_dir}.")

    dst_dir = os.path.join(src_dir + "_" + subset_name)
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    image_list = sorted(glob.glob(os.path.join(src_dir, "*.png")))

    return image_list, dst_dir


def crop(length, l_split, overlap):

    if length <= l_split:
        position = [(0, length)]
        n_crop = 1
    elif overlap != 0:
        px_overlap = int(l_split * overlap)
        start_next = 0
        n_crop = int((length - px_overlap) / (l_split - px_overlap))
        position = list(range(n_crop))
        for i_crop in range(n_crop):
            start = start_next
            end = start + l_split
            start_next = end - px_overlap
            position[i_crop] = (start, end)
    else:
        n_crop, rem = divmod(length, l_split)
        if rem == 0:
            px_overlap = 0
            err = 0
        else:
            px_overlap = (l_split - rem) // n_crop
            sum_crop = (l_split - px_overlap) * n_crop + l_split
            err = sum_crop - length
            n_crop += 1
        position = list(range(n_crop))
        start_next = 0
        for i_crop in range(n_crop):
            start = start_next
            end = start + l_split
            start_next = end - px_overlap
            if i_crop < err:
                start_next -= 1
        # print(i_h, left, right, left_next)
            position[i_crop] = (start, end)
      # print(pos_h)
    return position


def resize(input_size, input_res, split_size, split_tile):
    target_size = split_tile / split_size
    factor = input_res / target_size
    resize_size = round(input_size * factor)
    return resize_size


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='input')
    parser.add_argument('--target_dir', type=str, default='target')
    parser.add_argument('--split_h', type=int, default=256)
    parser.add_argument('--split_w', type=int, default=256)
    parser.add_argument('--split_name', type=str, default='split')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--resize_name', type=str, default='resize')
    parser.add_argument('--input_res_h', type=float, default=1.0)
    parser.add_argument('--input_res_w', type=float, default=1.0)
    parser.add_argument('--split_tile_h', type=float, default=1000)
    parser.add_argument('--split_tile_w', type=float, default=1000)
    parser.add_argument('--exclude', action='store_true')
    parser.add_argument('--exclude_th', type=float, default=0)
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--num_start', type=int, default=0, choices=[0, 1])
    parser.add_argument('--polygon_split', type=int, default=1, help='Descripción del argumento polygon_split')
    parser.add_argument('--dummy_ratio', type=int, default=2, help='Descripción del argumento dummy_ratio')
    parser.add_argument('--shift8', action='store_true', help='Descripción del argumento shift8')
    parser.add_argument('--border_exclude', action='store_true', help='Descripción del argumento border_exclude')
    parser.add_argument('--inflation', type=int, default=1, help='Descripción del argumento inflation')
    parser.add_argument('--lower_th', type=float, default=0.05, help='Descripción del argumento lower_th')
    parser.add_argument('--upper_th', type=float, default=0.8, help='Descripción del argumento upper_th')
    parser.add_argument('--split_record', type=str, default='cb_vss_train_4x/split_record.json', help='Descripción del argumento split_record')


    FLAGS = parser.parse_args()

    h_split = FLAGS.split_h
    w_split = FLAGS.split_w

    input_list, input_split_dir = preparation(FLAGS.input_dir, FLAGS.split_name)
    target_list, target_split_dir = preparation(FLAGS.target_dir, FLAGS.split_name)
    if FLAGS.resize:
        _, input_resize_dir = preparation(FLAGS.input_dir, FLAGS.resize_name)
        _, target_resize_dir = preparation(FLAGS.target_dir, FLAGS.resize_name)

    for input_file, target_file in zip(input_list, target_list):
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        target_name = os.path.splitext(os.path.basename(target_file))[0]
        # print(input_name, target_name)

        img1 = cv2.imread(input_file)
        img2 = cv2.imread(target_file)

        if img1.shape != img2.shape:
            raise ValueError("The size of input {} does not match with that of target {}.".format(input_name, target_name))

        if FLAGS.resize:
            dsize_h = resize(img1.shape[0], FLAGS.input_res_h, h_split, FLAGS.split_tile_h)
            dsize_w = resize(img1.shape[1], FLAGS.input_res_w, w_split, FLAGS.split_tile_w)
            # print(img1.shape)
            # print(dsize_h, dsize_w)

            img1_r = cv2.resize(img1, (dsize_w, dsize_h))
            img2_r = cv2.resize(img2, (dsize_w, dsize_h))

            # if FLAGS.overlap == 0:
            #     img1_s = img1_r
            #     img2_s = img2_r
            # else:
            #     n_h_o, rem_h_o = divmod(dsize_w, w_split)
            #     margin_L = -(- rem_h_o // 2)
            #     margin_R = dsize_w - (rem_h_o - margin_L)

            #     n_v_o, rem_v_o = divmod(dsize_h, h_split)
            #     margin_T = -(- rem_v_o // 2)
            #     margin_B = dsize_h - (rem_v_o - margin_T)
                
            #     img1_s = img1_r[margin_T:margin_B, margin_L:margin_R, :]
            #     img2_s = img2_r[margin_T:margin_B, margin_L:margin_R, :]

            img1_s = img1_r
            img2_s = img2_r

            input_resize = os.path.join(input_resize_dir, os.path.basename(input_file))
            target_resize = os.path.join(target_resize_dir, os.path.basename(target_file))
            cv2.imwrite(input_resize, img1_s)
            cv2.imwrite(target_resize, img2_s)
            # print(img1_s.shape, img2_s.shape)
        else:
            img1_s = img1
            img2_s = img2

        # sys.exit()

        # h1, w1 = img1.shape[:2]
        # h2, w2 = img2.shape[:2]
        height, width = img1_s.shape[:2]
        # px_overlap_h = h_split * FLAGS.overlap

        pos_h = crop(width, w_split, FLAGS.overlap)
        pos_v = crop(height, h_split, FLAGS.overlap)
        # print(width, height, n_h, n_v, rem_h, rem_v)
        # print(overlap_h, overlap_v, sum_h, sum_v, err_h, err_v)

        for i_v in range(len(pos_v)):
            for i_h in range(len(pos_h)):
                # print(i_h, i_v, pos_h[i_h], pos_v[i_v], pos_h[i_h][0], pos_h[i_h][1], pos_v[i_v][0], pos_v[i_v][1])
                img1_split = img1_s[pos_v[i_v][0]:pos_v[i_v][1], pos_h[i_h][0]:pos_h[i_h][1], :]
                img2_split = img2_s[pos_v[i_v][0]:pos_v[i_v][1], pos_h[i_h][0]:pos_h[i_h][1], :]

                if FLAGS.exclude:
                    if np.all(img2_split == 255):
                        continue

                    if FLAGS.exclude_th != 0:
                        if (np.count_nonzero(np.all(img2_split != [255,255,255], axis=2)) < FLAGS.exclude_th * h_split * w_split):
                            continue
                
                i_v_n = i_v + FLAGS.num_start
                i_h_n = i_h + FLAGS.num_start
                split_name = "{}_{:02d}_{:02d}.png".format(input_name, i_v_n, i_h_n)
                input_split = os.path.join(input_split_dir, split_name)
                cv2.imwrite(input_split, img1_split)
                target_split = os.path.join(target_split_dir, split_name)
                cv2.imwrite(target_split, img2_split)
    
    # Agregar el código para guardar el registro en un archivo JSON
    split_records = {
        'input_files': input_list,
        'target_files': target_list,
        # Agrega aquí cualquier otro dato que desees registrar
    }

    # Ruta del archivo JSON
    split_record_path = FLAGS.split_record

    # Guardar el diccionario en el archivo JSON
    with open(split_record_path, 'w') as json_file:
        json.dump(split_records, json_file, indent=4)


if __name__ == "__main__":
    main()