import numpy as np 
from functions.others import * 
from tqdm import tqdm, tqdm_notebook
import pandas as pd 
import time 

def sparse_bin(features, bin_num, randomize=True, fault=None):
    err = np.random.randn(features.shape[1]) * randomize
    cat = np.zeros(features.shape[0]).astype('int64')
    factore = 1
    for i, feat in enumerate(features.columns):
        cat += (features[feat] * bin_num._asdict()[feat] + err[i]).astype('int64') * factore
        factore *= 2 * bin_num._asdict()[feat] + 1
    if fault is not None:
        cat += (factore * features.index * fault).astype('int64')
    return tag_bins(cat)


def clustering(hits, stds, filters, phik=3.3, nu=250):
    rest = hits.copy()
    rest['track_len'] = 1
    rest['track_id'] = -rest.index
    rest['kt'] = 1e-6
    rest['z0'] = 0

    rest['sensor'] = rest.volume_id + 100*rest.layer_id + 100000*rest.module_id
    rest['layer'] = rest.volume_id + 100*rest.layer_id
    
    maxprog = filters.npoints.sum()
    rest['pre_track_id'] = rest.track_id
    
    label_shift_M=1000000
    p = -1

    weights={'phi':1, 'theta':0.15}
    res_list = []
    pbar = tqdm(total=maxprog,mininterval=5.0)

    for filt in filters.itertuples():
        test_points = pd.DataFrame()
        for col in stds:
            test_points[col] = np.random.normal(scale=stds[col], size=filt.npoints)
        for row in test_points.itertuples():
            pbar.update()
            p += 1
            calc_features(rest, row, phik)
            feat = ['phi', 'sint', 'cost']
            rest['new_track_id'], rest['new_track_len'] = sparse_bin(rest[feat], filt, fault=rest.fault)
            rest['new_track_id'] += (p + 1) * label_shift_M
            better = (rest['new_track_len'] > rest['track_len']) & (rest['new_track_len'] < 19)
            rest['new_track_id'] = rest['new_track_id'].where(better, rest['track_id'])
            dum, rest['new_track_len'] = tag_bins(rest['new_track_id'])
            better = (rest['new_track_len'] > rest['track_len']) & (rest['new_track_len'] < 19)
            rest['track_id'] = rest['track_id'].where(~better, rest['new_track_id'])
            rest['track_len'] = rest['track_len'].where(~better, rest['new_track_len'])
            rest['kt'] = rest['kt'].where(~better, row.kt)
            rest['z0'] = rest['z0'].where(~better, row.z0)
            
            if (row.Index + 1) % nu == 0 or (row.Index + 1) == test_points.shape[0]:
                # outlier removal
                dum, rest['track_len'] = tag_bins(rest['track_id'])
                calc_features(rest, rest[['kt', 'z0']], phik)
                
                gp = rest.groupby('track_id').agg({
                    'phi': np.mean, 'sint': np.mean, 'cost': np.mean
                }).rename(columns={'phi': 'mean_phi', 'sint': 'mean_sint', 'cost': 'mean_cost'}).reset_index()
                cols_drop = rest.columns.intersection(gp.columns).drop('track_id')
                rest = rest.drop(cols_drop, axis=1).reset_index().merge(gp, on='track_id', how='left').set_index('index')
                
                rest['dist'] = (weights['theta'] * np.square(rest['sint'] - rest['mean_sint']) +
                            weights['theta'] * np.square(rest['cost'] - rest['mean_cost']) +
                            weights['phi'] * np.square(rest['phi'] - rest['mean_phi']))
                rest = rest.sort_values('dist')
                rest['closest'] = rest.groupby(['track_id', 'sensor'])['dist'].cumcount()
                rest['closest2'] = rest.groupby(['track_id', 'layer'])['dist'].cumcount()
                select = (rest['closest'] != 0) | (rest['closest2'] > 2)
                rest['track_id'] = rest['track_id'].where(~select, rest['pre_track_id'])
                dum, rest['track_len'] = tag_bins(rest['track_id'])
                
                select = rest['track_len'] > filt.min_group
                tm = rest[select][['hit_id', 'track_id', 'kt', 'z0']]
                res_list.append(tm)
                rest = rest[~select]
                dum, rest['track_len'] = tag_bins(rest['track_id'])
                rest['pre_track_id'] = rest['track_id'] 

    res_list.append(rest[['hit_id', 'track_id', 'kt', 'z0']].copy())
    res = pd.concat(res_list, ignore_index=True)
    pbar.close()
    rest['track_id'],dum=tag_bins(rest['track_id'])
    return res 
