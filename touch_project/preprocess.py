import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def preprocess(data_dir: Path, target_dir: Path, animal: int):
    for animal_dir in data_dir.iterdir():
        if "Touch" not in str(animal_dir):
            continue

        target_animal_dir = target_dir / animal_dir.name
        target_animal_dir.mkdir(exist_ok=True)

        spike_data = np.load(animal_dir / "spks_final.npy")
        inputs = spike_data.transpose()

        # Hard-coding this loading procedure, but in the future
        # it would be good to save data with named columns
        touch_data = np.load(animal_dir / "touch_behav.npy")
        starts = touch_data[:, 0]
        ends = touch_data[:, 1]
        limbs = touch_data[:, 5]

        coord_conversion = np.load(animal_dir / "idx_coord_neural.npy")
        starts_converted = coord_conversion[starts.astype(int)]
        # Substract 1 because the final end time is the same as the
        # length of the coordinate conversion array, so I interpret
        # the end points as the first index that touching ceased
        ends_converted = coord_conversion[ends.astype(int) - 1]
        targets = np.zeros(max(coord_conversion) + 1)
        for start, end, limb in zip(starts_converted, ends_converted, limbs):
            targets[start : end + 1] += int(limb)

        inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
        targets = torch.from_numpy(targets).type(torch.FloatTensor)
        print(f"Animal {animal} neural signal shape: {inputs.shape}")
        print(f"Animal {animal} limb moved shape: {targets.shape}")

        with open(target_animal_dir / "inputs_pickle.pkl", "wb") as f:
            pickle.dump(inputs, f)

        with open(target_animal_dir / "targets_pickle.pkl", "wb") as g:
            pickle.dump(targets, g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess neural data")
    parser.add_argument("data_dir", type=Path, help="Path to raw data")
    parser.add_argument(
        "target_dir", type=Path, help="Path where pickle files will be written"
    )
    args = parser.parse_args()

    args.target_dir.mkdir(parents=True, exist_ok=True)

    preprocess(args.data_dir, args.target_dir, 1)
