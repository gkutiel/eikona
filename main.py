import numpy as np
import pandas as pd

def read_train():
    total_rows = 32_343_174
    n = 10_000

    # Randomly choose rows to skip
    skip = sorted(np.random.choice(np.arange(1, total_rows+1), 
                                    size=total_rows-n, 
                                    replace=False))    