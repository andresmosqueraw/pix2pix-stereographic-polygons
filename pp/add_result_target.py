
from PIL import Image
import numpy as np
from glob import glob
import argparse
from tqdm import tqdm
import os
import shutil

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type = str, default = "./results/jisuberi2_pix2pix/test_latest/images")
    parser.add_argument("--dataroot", type = str, default = "./datasets/jisuberi2")
    # parser.add_argument("--mode", type = str, default = "add", help="add: add target to result_dir. concat: concat 3 images from result_dir")
    args = parser.parse_args()
    return args

def main(args):
    image_list_fake = glob(args.results_dir + "/*_fake.png") 
    os.makedirs(args.results_dir + "_tf", exist_ok = True)
    for fake_path_org in tqdm(image_list_fake):
        fake_path = fake_path_org.replace("/images\\", "/images_tf\\")
        # if args.mode == "add":
        fake_name = fake_path.split("/")[-1].split("\\")[-1]
        image_path = args.dataroot + "/test/" + fake_name.replace("_fake.", ".")
        image = Image.open(image_path)
        image= image.resize((512, 256))
        image_array = np.array(image)
        image = Image.fromarray(image_array[:, 256:, :])
        target_path = fake_path.replace("_fake.", "-targets.")
        image.save(target_path)

        shutil.copy(fake_path_org, fake_path.replace("_fake.", "-outputs."))
        shutil.copy(fake_path_org.replace("_fake.", "_real."), fake_path.replace("_fake.", "-inputs."))

        # elif args.mode == "concat":
        #     real_path = fake_path.replace("_fake", "_real")
        #     target_path = fake_path.replace("_fake", "_target")
        #
        #     fake_image = Image.open(fake_path)
        #     real_image = Image.open(real_path)
        #     target_image = Image.open(target_path)
        #     fake_array = np.array(fake_image)
        #     real_array = np.array(real_image)
        #     target_array = np.array(target_image)
        #
        #     image_array = np.concatenate([real_image, fake_image, target_image], 1)
        #     image = Image.fromarray(image_array)
        #     image.save(args.results_dir.rsplit("/", 1)[0] + "_concatenate/" + fake_path.split("/")[-1].split("\\")[-1].replace("_fake", "_rft"))
        #
        # else:
        #     print("mode error: mode is add(default) or concat")
            # exit()

if __name__ == "__main__":
    args = arg_parse()
    main(args)
