"""extract_patches.py

Patch extraction script.
"""
import re
import glob
import os
import tqdm
import pathlib
import argparse
import numpy as np
import pdb

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir
from dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate patches')
    parser.add_argument('--patch_dir', type=str, default="sample_data/patch", required=False, \
                        help='file path to read the actual patch images')
    parser.add_argument('--dataset_class', type=str, default="orion_GMM_7_class", required=False, \
                        help='file path to read the actual patch images')
    parser.add_argument('--mat_dir', type=str, default="sample_data/mat", required=False, \
                    help='file path to read and save the selected patches')
    parser.add_argument('--save_dir', type=str, default="output/training_data", required=False, \
                        help='file path to read and save the selected patches')
    args = parser.parse_args()

    print("=============================Args for this experiment=====================================")
    print(args)
    print("==========================================================================================")

    source_path = "."
    patch_dir = os.path.join(source_path, args.patch_dir)
    mat_dir = os.path.join(source_path, args.mat_dir)
    save_dir = os.path.join(source_path, args.save_dir)

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_class = args.dataset_class
    print(f'Dataset class: {dataset_class}')
    
    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True
    win_size = [540, 540]
    step_size = [164, 164]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Load pre-selected patches for all samples
    unique_samples = ['CRC08']
    print(f'Found {len(unique_samples)} unique samples')
    selected_samples = unique_samples.copy()

    for sample_id in selected_samples:
        print(f'Processing {sample_id}')

        image_path = os.path.join(patch_dir)
        ann_path = os.path.join(mat_dir)

        # a dictionary to specify where the dataset path should be
        dataset_info = {"img": (".tiff", image_path), "ann": (".mat", ann_path)}

        patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
        parser = get_dataset(dataset_class)
        xtractor = PatchExtractor(win_size, step_size)

        img_ext, img_dir = dataset_info["img"]
        ann_ext, ann_dir = dataset_info["ann"]

        save_root = os.path.join(save_dir, sample_id)
        out_dir = "%s/%dx%d_%dx%d/" % (
            save_root,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )

        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform

        # Perform some filtering based on selected patches
        #file_list = [file_path for file_path in file_list if pathlib.Path(file_path).stem in selected_patches]

        rm_n_mkdir(out_dir)
        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )

            # img (1000, 1000, 3)
            # ann (1000, 1000, 2) ("inst_map", "type_map") 

            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch) # patch(540, 540, 5)
                pbar.update()
            pbar.close()
            # *
            
            pbarx.update()
        pbarx.close()
