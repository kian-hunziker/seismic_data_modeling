import re
import os
import glob
import numpy as np
import torch


def get_metadata(f_name: str) -> dict:
    f_name = f_name.replace('..', '.')

    meta = f_name.split('.')
    station = meta[1]
    channel = meta[2]
    year = meta[3][:4]
    day = meta[3][4:7]
    meta = {
        'station': station,
        'channel': channel,
        'year': year,
        'day': day
    }
    return meta


def format_label(s):
    """
    Convert strings: 'processed_i4.HDC3.HHZ.2022287_0+.pt'->'HDC3.HHZ.2022287'.
    """
    # Remove leading parts before the first dot (and any extra dots)
    after_first_dot = s.split('.', 1)[1]
    # Remove the trailing parts after the last dot
    before_last_dot = re.split(r'\.[^\.]+$', after_first_dot)[0]
    # Replace multiple dots with a single dot, if needed
    single_dots = re.sub(r'\.+', '.', before_last_dot)
    return single_dots.split('_')[0]


def find_data_min_and_max(filename: str) -> tuple:
    file_list = glob.glob(os.path.join(filename, '*.pt'))

    d_min = torch.inf
    d_max = - torch.inf
    for f in file_list:
        data = torch.load(f)
        temp_min = torch.min(data)
        temp_max = torch.max(data)
        if temp_max > d_max:
            d_max = temp_max
        if temp_min < d_min:
            d_min = temp_min
    return d_min, d_max


if __name__ == '__main__':
    path = '../data/costa_rica/small_subset'
    d_min, d_max = find_data_min_and_max(path)
    print(f'd_min: {d_min}, d_max: {d_max}')
