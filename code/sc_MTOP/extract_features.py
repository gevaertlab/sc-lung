import os
from F3_FeatureExtract import fun3
import pandas as pd
import argparse
import pickle
import pdb

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--start', type=int, default=None,
                        help='Index for start slide')
    parser.add_argument('--end', type=int, default=None,
                        help='Index for ending slide')

    args = parser.parse_args()

    source_dir = "."
    wsi_dir = os.path.join(source_dir, "sample_data/pathology")
    hovernet_dir = os.path.join(source_dir, "sample_data")
    hovernet_dataset = "orion_GMM_7_class"
    save_path = os.path.join(hovernet_dir, 'mtop')
    os.makedirs(save_path, exist_ok = True)

    samples = sorted([f.split('.')[0] for f in os.listdir(os.path.join(hovernet_dir, "json"))])

    exist_samples = []
    for sample in samples:
        if os.path.exists(os.path.join(save_path, sample, f"{sample}_Edges.csv")):
            exist_samples.append(sample)
    print(f"Number of slides already processed: {len(exist_samples)}")
    samples = list(set(samples) - set(exist_samples))
    print(f"Number of slides to process: {len(samples)}")

    # indexing based on values for parallelization
    if args.start is not None and args.end is not None:
        samples = samples[args.start:args.end]
    elif args.start is not None:
        samples = samples[args.start:]
    elif args.end is not None:
        samples = samples[:args.end]
    print(f'Number of slides in current batch: {len(samples)}')
    print(f"Current batch: {samples}")

    for sample in samples:
        json_path = os.path.join(hovernet_dir, 'json',  f"{sample}.json")
        wsi_path = os.path.join(wsi_dir, f"{sample}.tiff")
        if os.path.exists(os.path.join(save_path, sample, f"{sample}_Edges.csv")):
            print(f"Feature already extracted for {sample}, skipped")
            continue
        try:
            print(f"Processing {sample}")
            fun3(json_path, wsi_path, save_path, n_process = 4,\
                 dataset = hovernet_dataset, cal_morph = True, cal_texture = True, cal_graph=True)
        except Exception as e:
            print(sample)
            print(e)
            continue