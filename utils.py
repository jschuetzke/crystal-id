import numpy as np

def map_points_to_steps(positions, intensities, steps):
    step_size  = np.mean(np.diff(steps))
    """ Maps calculated peak positions and intensities to steps of scan

    Inputs:
        positions [list, float]: peak positions
        intensities [list, float]: peak intensities
        steps np.array: steps for measurements

    Returns:
        signal np.array: mapped peaks to defined steps
    """    
    signal = np.zeros_like(steps)
    for i in range(positions.size):
        pos = positions[i]
        # get the step closest to the simulated positions
        diffs = np.argsort(np.abs(pos - steps))
        # angles in steps increasing so lower datapoint has lower index
        lower, upper = sorted(diffs[:2])
        # calculate ratio to split intensity
        ratio = (steps[upper] - pos) / step_size
        # ratio is "lever" -> apply inversely
        signal[lower] = ratio * intensities[i]
        signal[upper] = (1 - ratio) * intensities[i]
    return signal

def scale_min_max(ndarray, clip_perc=False, perc=0.2, 
                  output_max=False, input_max=None):
    x = ndarray.copy()
    if clip_perc:
        if x.ndim == 1:    
            pc = np.percentile(x, perc)
            x = np.clip(x, pc, None)
        else:
            pc = np.percentile(x, perc, axis=1)
            x = np.clip(x, np.expand_dims(pc, -1), None)
    max_arr = input_max
    if x.ndim == 1:
        min_arr = np.min(x, axis=0)
        if max_arr is None:
            max_arr = np.max(x, axis=0)
    else:
        min_arr = np.min(x, axis=1, keepdims=True)
        if max_arr is None:
            max_arr = np.max(x, axis=1, keepdims=True)
    if output_max:
        return ((x - min_arr) / (max_arr - min_arr)), max_arr
    # deprecated
    # if x.ndim == 1:
    #     x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    # else:
    #     x = (x - x.min(axis=1, keepdims=True)) / \
    #         (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True))
    return (x - min_arr) / (max_arr - min_arr)
