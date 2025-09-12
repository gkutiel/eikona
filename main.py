
from heapq import heappop, heappush
from typing import cast

import pandas as pd

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


def uplift(segment: pd.DataFrame):
    ctr_base = segment[target].mean()

    group_ctrs = segment.groupby(decision)[target].mean()

    cat = group_ctrs.idxmax()
    max_ctr = group_ctrs[cat]

    return (max_ctr / ctr_base - 1), cat


def sub_segments(*,
                 segment: pd.DataFrame,
                 query: dict[str, int]):

    cols = features - set(query.keys())
    for col in cols:
        for v, g in segment.groupby(col):
            q = query.copy()
            q[col] = cast(int, v)
            yield g, q


def search():
    df = pd.read_csv('train.csv', nrows=10_000)

    # optimize memory
    for col in df.columns:
        df[col] = df[col].astype("category")

    heap = []
    query = {}
    uplift_base, cat = uplift(df)

    id = 0
    heappush(heap, (-uplift_base, id, df, query))

    best_uplift = uplift_base
    while heap:
        _, _, seg, q = heappop(heap)
        for sub_seg, q_new in sub_segments(
                segment=seg,
                query=q):

            if len(sub_seg) < len(df) * 0.1:
                # print('Skipping small segment:', len(sub_seg))
                continue

            # Found a new segment
            id += 1
            up, cat = uplift(sub_seg)
            if best_uplift < up:
                best_uplift = up
                print(q, f'Best Uplift: {up:.2%} by {cat}')

            heappush(heap, (-up, id, sub_seg, q_new))


if __name__ == "__main__":
    search()
