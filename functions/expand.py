import pandas as pd 
import numpy as np 
from tqdm import tqdm
from functions.others import *

def refine_hipos(res, hits, stds, nhipo, weights):
    cols = list(res.columns)
    groups = res.merge(hits, on='hit_id', how='left')
    calc_features(groups, groups[['kt', 'z0']])
    gp = groups.groupby('track_id').agg({'phi': np.std, 'sint': np.std, 'cost': np.std}).reset_index()
    gp = gp.rename(columns={'phi': 'phi_std', 'sint': 'sint_std', 'cost': 'cost_std'})
    groups = groups.merge(gp, on='track_id', how='left')
    groups['theta_std'] = np.sqrt(weights['theta'] * np.square(groups['sint_std']) +
                                    weights['theta'] * np.square(groups['cost_std']))
    hipos = pd.DataFrame()
    
    for col in stds:
        hipos[col] = np.random.normal(scale=stds[col], size=nhipo)
    
    for hipo in tqdm(hipos.itertuples(), total=nhipo):
        groups['kt_new'] = groups['kt'] + hipo.kt 
        groups['z0_new'] = groups['z0'] + hipo.z0
        calc_features(groups, groups[['kt_new', 'z0_new']].rename(columns={'kt_new':'kt', 'z0_new':'z0'}))
        gp = groups.groupby('track_id').agg({'phi': np.std, 'sint': np.std, 'cost': np.std})
        gp = gp.rename(columns={'phi': 'new_phi_std', 'sint': 'new_sint_std', 'cost': 'new_cost_std'}).reset_index()
        groups = groups.merge(gp, on='track_id', how='left')
        groups['new_theta_std'] = np.sqrt(weights['theta'] * np.square(groups['new_sint_std']) +
                                    weights['theta'] * np.square(groups['new_cost_std']))
        old_std = np.sqrt(np.square(groups['theta_std']) + weights['phi'] * np.square(groups['phi_std']))
        new_std =  np.sqrt(np.square(groups['new_theta_std']) + weights['phi'] * np.square(groups['new_phi_std']))
        cond = old_std <= new_std 
        groups['kt'] = groups['kt'].where(cond, groups['kt_new'])
        groups['z0'] = groups['z0'].where(cond, groups['z0_new'])
        groups['theta_std'] = groups['theta_std'].where(cond, groups['new_theta_std'])
        groups['phi_std'] = groups['phi_std'].where(cond, groups['new_phi_std'])
        groups['sint_std'] = groups['sint_std'].where(cond, groups['new_sint_std'])
        groups['cost_std'] = groups['cost_std'].where(cond, groups['new_cost_std'])
        groups = groups.drop(['new_theta_std', 'new_phi_std', 'new_cost_std', 'new_sint_std'], axis=1)
        
    return groups[cols]
    

def expand_tracks(res, hits, min_track_len, max_track_len, max_expand, to_track_len,
                  mstd=1, dstd=0, max_dtheta=10, mstd_size=None, mstd_vol=None, drop=0, nhipo=1000, weights=None):
    if weights is None:
        weights = {'theta': 0.1, 'phi': 1.0}
        
    gp = res.groupby('track_id').first().reset_index()
    orig_hipo = gp[['track_id', 'kt', 'z0']]
    eres = res.copy()
    res_list = []
    stds = {'kt': 7e-5, 'z0': 0.8}
    eres = refine_hipos(eres, hits, stds, nhipo, weights)
    _, eres['track_len'] = tag_bins(eres['track_id'])
    eres['max_track_len'] = np.clip(eres['track_len'] + max_expand, 0, max_track_len)
    eres = eres.sort_values('track_len')
    eres = eres.merge(hits, on='hit_id', how='left')
    eres['sensor'] = eres.volume_id + 100 * eres.layer_id + 100000 * eres.module_id
    group_sensor = eres.groupby('track_id')['sensor'].unique()
    groups = eres[eres.track_len > min_track_len].groupby('track_id').first().reset_index().copy()
    groups = groups.sort_values('track_len', ascending=False)
    select = eres.track_len < to_track_len
    grouped = eres[~select].copy()
    regrouped = eres[select].copy()
    regrouped['min_dist'] = 100
    regrouped['new_track_len'] = 0
    regrouped['new_track_id'] = regrouped['track_id']
    regrouped['new_kt'] = regrouped['kt']
    regrouped['new_z0'] = regrouped['z0']
    regrouped['new_max_size'] = max_track_len
    
    for group_tul in tqdm(groups.itertuples(), total=groups.shape[0]):
        if group_tul.track_len >= max_track_len: continue
        group = eres[eres.track_id == group_tul.track_id].copy()
        calc_features(group, group[['kt', 'z0']])
        group['abs_z'] = np.abs(group.z)
        group['abs_theta'] = np.abs(group.theta)
        phi_mean = group.phi.mean()
        sint_mean = group.sint.mean()
        cost_mean = group.cost.mean()
        max_z = group.abs_z.max()
        max_theta = group.abs_theta.max()
        
        dtheta = np.abs(group.theta.max() - group.theta.min())
        
        regrouped['abs_z'] = np.abs(regrouped.z)
        calc_features(regrouped, group_tul, double_sided=True)
        regrouped['dist'] = np.sqrt(weights['theta'] * np.square(regrouped.sint - sint_mean) +
                                    weights['theta'] * np.square(regrouped.cost - cost_mean) +
                                    weights['phi'] * np.square(regrouped.phi - phi_mean))
        regrouped['dist2'] = np.sqrt(weights['theta'] * np.square(regrouped.sint2 - sint_mean) +
                                    weights['theta'] * np.square(regrouped.cost2 - cost_mean) +
                                    weights['phi'] * np.square(regrouped.phi2 - phi_mean))
        select = (regrouped.abs_z > max_z) & (regrouped.dist > regrouped.dist2) & (max_dtheta < max_dtheta)
        regrouped['dist'] = regrouped.dist.where(~select, regrouped.dist2)
        cmstd = regrouped.volume_id.map(mstd_vol) + mstd_size[group_tul.track_len] + mstd 
        
        sdsts = dstd 
        better = ((regrouped.dist < cmstd * sdsts) & (regrouped.dist < regrouped.min_dist) &
                    ~regrouped.sensor.isin(group_sensor.loc[group_tul.track_id]))
        regrouped['min_dist'] = regrouped.dist.where(better, regrouped.min_dist)
        regrouped['new_track_id'] = np.where(better, group_tul.track_id, regrouped.new_track_id)
        regrouped['new_z0'] = np.where(better, group_tul.z0, regrouped.new_z0)
        regrouped['new_kt'] = np.where(better, group_tul.kt, regrouped.new_kt)
        regrouped['new_track_len'] = np.where(better, group_tul.track_len, regrouped.new_track_len)
        regrouped['new_max_size'] = np.where(better, group_tul.max_track_len, regrouped.new_max_size)
        
    regrouped = regrouped.sort_values('min_dist')
    regrouped['closest'] = regrouped.groupby('new_track_id')['min_dist'].cumcount()
    better = regrouped.closest | (regrouped.new_track_len >= regrouped.new_max_size)
    regrouped['track_id'] = regrouped.track_id.where(better, regrouped.new_track_id)
    res_list.append(regrouped[['hit_id', 'track_id']])
    res_list.append(grouped[['hit_id', 'track_id']])
    to_return = pd.concat(res_list, ignore_index=True)   
    
    return to_return.merge(orig_hipo, on='track_id', how='left')