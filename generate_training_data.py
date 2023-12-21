import argparse
import os
import warnings
import yaml

import numpy as np
from functools import partial
from pymatgen.core import Structure
from scipy.ndimage import binary_dilation, gaussian_filter1d
from speedup import generate_varied_pattern
from multiprocessing import Pool
from tqdm import tqdm
from utils import map_points_to_steps
import wrapper
import background


def generate_structure_variation(structure_path, config, n):
    measurement_method = str.lower(config["domain"]["method"])
    measurement_range = config["measurement"]["range"]
    height_var = config["simulation"]["intensity_variation"]
    steps = np.arange(
        measurement_range[0], measurement_range[1], config["measurement"]["step"]
    )
    rng = np.random.default_rng(2023)

    struct = Structure.from_file(structure_path)

    sim_name = f"{measurement_method}_simulator"

    # try to import simulator
    try:
        sim = getattr(wrapper, sim_name)(measurement_range[0], measurement_range[1])
    except AttributeError:
        ValueError(
            f"There is a simulation wrapper missing for domain {measurement_method}"
        )
    except TypeError:
        if measurement_method == "xrd":
            sim = getattr(wrapper, sim_name)(
                measurement_range[0],
                measurement_range[1],
                config["domain"]["radiation"],
            )

    generate_pattern = partial(
        generate_varied_pattern,
        pymatgen_structure=struct,
        SimClass=sim,
        max_strain=config["simulation"]["lattice_variation"] / 2,
    )

    patterns_list = []

    with Pool() as pool:
        for result in tqdm(
            pool.imap_unordered(generate_pattern, list(range(n))),
            total=n,
        ):
            patterns_list.append(result)

    x = np.zeros([n, steps.size])
    for i in range(n):
        pos, his = patterns_list[i]
        his = np.array(
            [
                np.clip((h + rng.uniform(-height_var, height_var)), 0.01, 1.2)
                for h in his
            ]
        )
        x[i] = map_points_to_steps(pos, his, steps)
    return x


def add_holder_peak(steps, position, n, variation=20):
    rng = np.random.default_rng(2023)
    x = np.zeros([n, steps.size])
    pos = rng.integers(position - variation, position + variation, size=n)
    his = rng.uniform(0.0, 0.2, size=n)
    x[np.arange(n), pos] = his
    return x


def add_random(x, restricted_area=30, max_peaks=5, height_min=0.02, height_max=0.2):
    x_add = np.zeros_like(x)
    rng = np.random.default_rng(2023)
    height_min = max(0.15, height_min) # ensure additional peaks have atleast 2% intensity
    for i in range(x.shape[0]):
        cur_peak_pos = x[i].astype(bool)
        # dilate positions to avoid too much overlap between single and multiphase peaks
        restricted = binary_dilation(cur_peak_pos, iterations=restricted_area)
        restricted[:100] = True
        restricted[-100:] = True
        # restricted[:750] = True
        # restricted[2000:] = True
        elig_pos = np.arange(x.shape[1])[np.invert(restricted)]
        imp_number = rng.integers(1, max_peaks)
        # position of multiphase peaks
        add_pos = rng.choice(elig_pos, imp_number)
        add_his = np.random.uniform(height_min, height_max, size=imp_number)
        x_add[np.repeat(np.array([i]), imp_number), add_pos] = add_his
    return x_add


def add_noise(x, level_min=0.01, level_max=0.07):
    rng = np.random.default_rng(2023)
    noise_lvls = rng.uniform(level_min, level_max, size=x.shape[0])
    # add noise, clip extreme values
    gaus = 1 / 3 * np.clip(rng.normal(0, 1, x.shape), -3, 3)
    # next we shift the noise from -1 and 1 to 0 and 1
    gaus = (gaus * 0.5) + 0.5
    # scale noise
    gaus *= noise_lvls[:, None]
    return gaus


def main(system_name: str, structure_path: str):
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    if not os.path.exists(system_name):
        os.mkdir(system_name)
    else:
        raise ValueError("Directory already exists, please choose a different name")

    measurement_method = str.lower(config["domain"]["method"])
    measurement_range = config["measurement"]["range"]
    step_size = config["measurement"]["step"]
    holder = config["measurement"]["holder_position"]
    impurities = config["measurement"]["impurity_cutoff"]
    fwhm_max = config["simulation"]["fwhm"]
    noise_max = config["simulation"]["noise_ratio"]
    background_max = config["simulation"]["background_ratio"]
    rng = np.random.default_rng(2023)

    steps = np.arange(measurement_range[0], measurement_range[1], step_size)

    holder_pos = np.argmin(np.abs(holder - steps)) if holder else None

    # crystal structure and variations
    n_total = config["simulation"]["n_patterns"]
    n_positive = n_total // 2
    n_negative = n_total - n_positive

    print("#### Generating positive examples ####")

    x_pos = generate_structure_variation(structure_path, config, n_positive)
    hold = x_pos.copy()
    if holder_pos:
        x_pos += add_holder_peak(steps, holder_pos, n_positive)

    if impurities >= 1:
        x_pos += add_random(x_pos, height_max=5., max_peaks=3)
    elif impurities > 0:
        x_pos += add_random(x_pos, height_max=impurities, max_peaks=3)

    kernels = rng.uniform(5.0, fwhm_max, size=n_positive)

    for i in range(n_positive):
        x_pos[i] = gaussian_filter1d(x_pos[i], kernels[i] / (2.355), mode="constant")
        x_pos[i] /= np.max(x_pos[i])

    x_pos += background.chebyshev(x_pos.shape, factor=background_max)
    x_pos += add_noise(x_pos, level_max=noise_max)

    y_pos = np.ones([n_positive])

    print("#### Generating negative examples ####")

    # choices
    # -> 0 only background (amorphous)
    # -> 1 alternative pattern
    # if structure is always major phase (impurities < 1)
    # -> 2 structure as impurity phase
    # if structure is always major phase (impurities = 1)
    # -> 3 structure plus impurities

    choice = rng.choice(4, size=n_negative, p=[0.1,0.2,0.2,0.5])#[0.1, 0.3, 0.3, 0.3])
    x_neg = np.zeros([n_negative, steps.size])

    # choice 0
    ind_0 = np.where(choice == 0)[0]
    x_neg[ind_0] += add_random(x_neg[ind_0], height_max=0.2, max_peaks=3)

    # choice 1
    ind_1 = np.where(choice == 1)[0]
    x_neg[ind_1] += add_random(
        x_neg[ind_1], height_max=1.0, height_min=0.2, max_peaks=10
    )

    # choice 2
    ind_2 = np.where(choice == 2)[0]
    x_neg[ind_2] += add_random(
        x_neg[ind_2], height_max=1.0, height_min=0.2, max_peaks=10
    )
    random_struct = rng.integers(n_positive, size=ind_2.size)
    factors = rng.uniform(0.05, 0.6, size=ind_2.size)
    x_neg[ind_2] += hold[random_struct] * factors[:, None]

    # choice 3
    ind_3 = np.where(choice == 3)[0]
    x_neg[ind_3] += add_random(
        x_neg[ind_3], height_max=0.2, height_min=impurities, max_peaks=2
    )
    random_struct = rng.integers(n_positive, size=ind_3.size)
    x_neg[ind_3] += hold[random_struct]

    if holder_pos:
        x_neg += add_holder_peak(steps, holder_pos, n_negative)
    kernels = rng.uniform(5.0, fwhm_max, size=n_positive)

    for i in range(n_negative):
        x_neg[i] = gaussian_filter1d(x_neg[i], kernels[i] / (2.355), mode="constant")
        if not i in ind_0.tolist():
            x_neg[i] /= np.max(x_neg[i])

    bkg = background.chebyshev(x_neg.shape, factor=background_max)
    bkg[ind_0] = background.decay([ind_0.size, steps.size])
    x_neg += bkg
    x_neg += add_noise(x_neg, level_max=noise_max)

    y_neg = np.zeros([n_negative])

    x = np.vstack([x_pos, x_neg])
    y = np.hstack([y_pos, y_neg])

    x /= np.max(x, axis=1, keepdims=True)

    n = x.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    train_val_split = int(n*0.8)
    x_train = x[indices[:train_val_split]]
    x_val = x[indices[train_val_split:]]
    y_train = y[indices[:train_val_split]]
    y_val = y[indices[train_val_split:]]

    np.save(os.path.join(system_name, "x_train.npy"), x_train)
    np.save(os.path.join(system_name, "x_val.npy"), x_val)

    np.save(os.path.join(system_name, "y_train.npy"), y_train)
    np.save(os.path.join(system_name, "y_val.npy"), y_val)

    np.save(os.path.join(system_name, "steps.npy"), steps)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate xrd signals")
    parser.add_argument(
        "system_name", type=str, nargs="?", help="name of the system to simulate"
    )
    parser.add_argument(
        "--structure_path",
        type=str,
        help="path to cif file containing structure",
    )

    args = parser.parse_args()
    main(**vars(args))
