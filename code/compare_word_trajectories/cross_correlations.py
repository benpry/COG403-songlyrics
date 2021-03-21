"""
This file contains the helper functions that compute cross correlations
"""
import numpy as np

def time_lag_cross_corrs(s1, s2):
    """
    s1 and s2 are pd.Series objects that we are computing the cross correlations of
    """
    rs = []
    for lag in range(-len(s1) // 2, len(s2) // 2):
        r = s1.corr(s2.shift(lag))
        rs.append(r)

    return np.array(rs)
