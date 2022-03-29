import numpy as np


def create_spiral(spiral_freq: float = 1,
                  spiral_size: float = 10,
                  start_point_range: float = 30,
                  noise_level: float = 0.1,
                  n_spiral: int = 500,
                  n_routs: int = 100):
    # make spiral
    t = np.linspace(0, spiral_size, n_spiral)
    x = t * np.sin(spiral_freq * t)
    y = t * np.cos(spiral_freq * t)

    start_point = np.random.uniform(spiral_size, start_point_range, 2)
    sign = np.random.choice([-1, 1], 2)
    start_point *= sign

    spiral_start = np.array([x[0], y[0]])
    spiral_end = np.array([x[-1], y[-1]])

    # make lines
    t = np.tile(np.linspace(1, 0, n_routs), reps=[2, 1]).T  # [N, 2]

    l1 = start_point * t + spiral_start * (1 - t)
    l2 = spiral_end * t + start_point * (1 - t)

    x = np.concatenate([l1[:, 0], x, l2[:, 0]])
    y = np.concatenate([l1[:, 1], y, l2[:, 1]])

    # add noise
    x += np.random.randn(len(x)) * noise_level
    y += np.random.randn(len(x)) * noise_level

    # make labels
    spiral_start_idx = l1.shape[0]
    spiral_end_idx = len(x) - l2.shape[0]

    return x, y, spiral_start_idx, spiral_end_idx


def generate_data(reps: int = 100,
                  spiral_freq_range=(1, 2.5),
                  spiral_size_range=(5, 10),
                  n_spiral_range=(200, 500),
                  n_routs_range=(50, 200),
                  start_point_range=30,
                  noise_level=0):
    data = []

    for i in range(reps):
        spiral_freq = np.random.uniform(*spiral_freq_range)
        spiral_size = np.random.uniform(*spiral_size_range)
        n_spiral = np.random.randint(*n_spiral_range)
        n_routs = np.random.randint(*n_routs_range)

        x, y, spiral_start_idx, spiral_end_idx = create_spiral(spiral_freq=spiral_freq,
                                                               spiral_size=spiral_size,
                                                               start_point_range=start_point_range,
                                                               noise_level=noise_level,
                                                               n_spiral=n_spiral,
                                                               n_routs=n_routs
                                                               )

        curr_data_dict = {'x': x,
                          'y': y,
                          'start': spiral_start_idx,
                          'end': spiral_end_idx}

        data.append(curr_data_dict)

    return data


def pad(x, y, start, end, target_len):
    """just repeat the first element"""

    len_diff = target_len - len(x)
    x = np.pad(x, (len_diff, 0), mode='edge')
    y = np.pad(y, (len_diff, 0), mode='edge')

    start = start + len_diff
    end = end + len_diff

    return x, y, start, end


def remove_samples(x, y, start, end, target_len):
    """randomly remove samples"""

    permuted_idxs = np.random.permutation(len(x))
    new_idxs = np.sort(permuted_idxs[:target_len])

    x = x[new_idxs]
    y = y[new_idxs]

    removed_idxs = permuted_idxs[target_len:]
    removed_before_start = np.sum(removed_idxs < start)
    removed_before_end = np.sum(removed_idxs < end)

    start = start - removed_before_start
    end = end - removed_before_end

    return x, y, start, end


def resize(x, y, start, end, target_len: int):
    """pad or cut to target len"""

    if len(x) < target_len:
        x, y, start, end = pad(x, y, start, end, target_len)

    elif len(x) > target_len:
        x, y, start, end = remove_samples(x, y, start, end, target_len)

    return x, y, start, end


def resize_data_list(data, target_len=600):
    new_data = []
    for sample in data:
        x, y, start, end = resize(**sample, target_len=target_len)
        curr_sample_dict = {'x': x,
                            'y': y,
                            'start': start,
                            'end': end}

        new_data.append(curr_sample_dict)

    return new_data
