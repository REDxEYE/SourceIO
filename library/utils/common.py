from typing import Sequence


def get_slice(data: Sequence, start, count=None):
    if count is None:
        count = len(data) - start
    return data[start:start + count]
