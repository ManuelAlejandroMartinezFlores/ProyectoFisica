import numpy as np
import pickle


def tag_bins(cat):
    un, bin_tag, cnt = np.unique(cat, return_inverse=True, return_counts=True)
    return bin_tag, cnt[bin_tag]

def calc_features(hits, hipos, phik=3.3, double_sided=False): # add phik
    if 'rr' not in list(hits.columns):
        hits['theta_'] = np.arctan2(hits.y, hits.x)
        hits['rr'] = np.sqrt(np.square(hits.y) + np.square(hits.x))
    ktrr = hipos.kt * hits.rr
    hits['dtheta'] = np.where(np.abs(ktrr) < 1, np.arcsin(ktrr, where=np.abs(ktrr)<1), ktrr)
    hits['theta'] = hits.theta_ + hits.dtheta
    hits['phi'] = np.arctan((hits.z - hipos.z0) * hipos.kt / (hits.dtheta * phik)) * 2.05 /np.pi
    hits['sint'] = np.sin(hits.theta)
    hits['cost'] = np.cos(hits.theta)
    hits['fault'] = (np.abs(ktrr) > 1).astype(np.int8)
    
    if double_sided:
        hits['phi2'] = np.arctan2(hits.z - hipos.z0, phik * (np.pi - hits.dtheta)) * 2 / np.pi 
        hits['theta2'] = hits.theta_ + np.pi - hits.dtheta 
        hits['sint2'] = np.sin(hits.theta2)
        hits['cost2'] = np.cos(hits.theta2)
    return hits
    
    
    
def score_event_fast(truth, submission):
    truth = truth[["hit_id", "particle_id", "weight"]].merge(submission, how='left', on='hit_id')
    df = truth.groupby(['track_id', 'particle_id']).hit_id.count().to_frame('count_both').reset_index()
    truth = truth.merge(df, how='left', on=['track_id', 'particle_id'])
    
    df1 = df.groupby(['particle_id']).count_both.sum().to_frame('count_particle').reset_index()
    truth = truth.merge(df1, how='left', on='particle_id')
    df1 = df.groupby(['track_id']).count_both.sum().to_frame('count_track').reset_index()
    truth = truth.merge(df1, how='left', on='track_id')
    truth.count_both *= 2
    score = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].weight.sum()
    particles = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].particle_id.unique()
    return score


def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    


def hit_score(res, truth):
    tt = res.merge(truth[['hit_id', 'particle_id', 'weight']], on='hit_id', how='left')
    _, tt['track_len'] = tag_bins(tt['track_id'])
    _, tt['real_track_len'] = tag_bins(tt['particle_id'])
    gp = tt.groupby('track_id')['particle_id'].value_counts().rename('par_freq').reset_index()
    tt = tt.merge(gp, on=['track_id', 'particle_id'], how='left')
    gp = gp.groupby(by='track_id').head(1)
    gp = gp.rename(index=str, columns={'particle_id': 'common_particle_id'})
    tt = tt.merge(gp.drop(['par_freq'], axis=1), on='track_id', how='left')
    tt['to_score'] = (2*tt['par_freq'] > tt['track_len']) & (2*tt['par_freq'] > tt['real_track_len'])
    tt['score'] = tt['weight'] * tt['to_score']
    return tt


def get_true_tracks(hits, particles, truth, min_hits = 4):
    hitst = hits.merge(truth[['hit_id', 'particle_id']], on='hit_id', how='left')
    hitst = hitst.merge(particles[['particle_id', 'nhits']])
    hitst = hitst[hitst.nhits > min_hits].rename(columns={'particle_id': 'track_id'})
    d = get_features(hitst)
    return d[['svolume', 'nhits', 'nclusters', 'nhitspercluster', 'xmax', 'ymax', 'zmax',
                'xmin', 'ymin', 'zmin', 'zmean', 'xvar', 'yvar', 'zvar']]
