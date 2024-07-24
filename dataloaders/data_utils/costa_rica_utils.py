import re


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
