from heapq import heappop, heappush
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import norm

target = 'click'
decision = 'site_category'
features = set([
    'banner_pos',
    'site_id',
    'site_domain',
    'app_id',
    'app_domain',
    'app_category',
    'device_id',
    # 'device_ip',
    'device_model',
    'device_type',
    'device_conn_type',
    'C1',
    'C14',
    'C15',
    'C16',
    'C17',
    'C18',
    'C19',
    'C20',
    'C21',
    'mday',
    'hour',
    'wday'])


def get_segment(df: pd.DataFrame, query: dict[str, int]):
    mask = pd.Series([True] * len(df))

    for k, v in query.items():
        mask &= df[k] == v

    return df[mask]


def pval(segment: pd.DataFrame, cat: str):
    """
    Compute p-value for the given segment.
    """
    return 0.0


MIN_CAT_SIZE = 30


def uplift(segment: pd.DataFrame):
    ctr_base = segment[target].mean()
    assert ctr_base > 0, "No clicks in segment"

    ups = [0]
    for cat in segment[decision].unique():
        mask = segment[decision] == cat
        if sum(mask) < MIN_CAT_SIZE or sum(~mask) < MIN_CAT_SIZE:
            continue

        ctr_new = segment[mask][target].mean()
        if ctr_new < ctr_base:
            continue

        ctr_rest = segment[~mask][target].mean()
        assert ctr_rest < ctr_new

        zc = (
            ctr_new - ctr_rest) / np.sqrt(
            ctr_base * (1 - ctr_base) * (1/sum(mask) + 1/sum(~mask)))

        pval = 2 * (1 - norm.cdf(abs(zc)))

        if pval < 0.05:
            ups.append(ctr_new / ctr_base - 1)

    return max(ups)


def sub_segments(
        *,
        segment: pd.DataFrame,
        query: dict[str, int]):

    cols = features - set(query.keys())
    for col in cols:
        for v, g in segment.groupby(col, observed=True):
            q = query.copy()
            q[col] = cast(int, v)
            yield g, q


def search():
    df = pd.read_csv('train.csv', nrows=10_000)
    df = df[[target, decision] + list(features)]

    # optimize memory
    for col in df.columns:
        if col == target:
            continue

        df[col] = df[col].astype("category")

    heap = []
    query = {}
    uplift_base = uplift(df)
    print('Base uplift:', uplift_base)

    id = 0
    heappush(heap, (-uplift_base, id, query))

    best_uplift = uplift_base
    for _ in range(100):
        _, _, q = heappop(heap)
        for sub_seg, q_new in sub_segments(
                segment=get_segment(df, q),
                query=q):

            if len(sub_seg) < len(df) * 0.1:
                # print('Skipping small segment:', len(sub_seg))
                continue

            # Found a new segment
            id += 1
            up = uplift(sub_seg)

            if best_uplift < up:
                print(q_new, f'Best Uplift: {up:.2%}')
                best_uplift = up

            heappush(heap, (-up, id, q_new))


if __name__ == "__main__":
    search()
