from sklearn.cluster import dbscan
import pandas as pd 
import numpy as np 
from functions.others import * 


def get_features(sub, cluster_size=10):
    hitst = sub.copy()
    X = np.column_stack([hitst.x.values, hitst.y.values, hitst.z.values, hitst.track_id.values*1000000])
    _, hitst['labels'] = dbscan(X, eps=cluster_size, min_samples=1, algorithm='ball_tree', metric='euclidean')
    gp = hitst.groupby('track_id').agg({'hit_id': 'count', 'labels': 'nunique', 'volume_id': 'min',
                                        'x': ['min', 'max', 'var'], 'y': ['min', 'max', 'var'], 'z': ['min', 'max', 'var', 'mean']})
    gp.columns = ["".join(t) for t in gp.columns.ravel()]
    gp = gp.rename(columns={'hit_idcount': 'nhits', 'labelsnunique': 'nclusters', 'volume_idmin': 'svolume'})
    gp['nhitspercluster'] = gp.nhits / gp.nclusters
    return gp


def get_predictions(sub, hits, model, min_len=4):
    sub = sub.merge(hits, on='hit_id', how='left')
    _, sub['nhits'] = tag_bins(sub['track_id'])
    sub_long = sub[sub['nhits'] >= min_len]
    sub_short = sub[sub['nhits'] < min_len]
    tracks_long = get_features(sub_long)
    columns=['svolume','nclusters', 'nhitspercluster', 'xmax','ymax','zmax', 'xmin','ymin','zmin', 'zmean',
         'xvar','yvar','zvar']
    tracks_long['quality'] = model.predict(tracks_long[columns].values)
    tracks_long = tracks_long.reset_index()
    tracks_short = sub_short.groupby('track_id')['nhits'].count().reset_index().drop('nhits', axis=1)
    tracks_short['quality'] = 0
    preds = pd.concat([tracks_long[['track_id' ,'quality']], tracks_short[['track_id' ,'quality']]], ignore_index=True)
    preds['quality'] = preds['quality'].fillna(1)
    return preds 

def merge_with_probas(sub1, sub2, pred1, pred2, len_factor=0):
    _, sub1['group_size'] = tag_bins(sub1['track_id'])
    _, sub2['group_size'] = tag_bins(sub2['track_id'])
    sub1 = sub1.merge(pred1, on='track_id', how='left')
    sub2 = sub2.merge(pred2, on='track_id', how='left')
    sub1['quality'] += len_factor * sub1['group_size']
    sub2['quality'] += len_factor * sub2['group_size']
    sub = sub1.merge(sub2, on='hit_id', suffixes=('', '_new'))
    mm = sub['track_id'].max() + 1
    sub['track_id_new'] += mm 
    cond = sub['quality'] >= sub['quality_new']
    
    for col in ['track_id', 'z0', 'kt']:
        sub[col] = sub[col].where(cond, sub[col+'_new'])
    
    sub = sub[['hit_id', 'track_id', 'event_id', 'kt', 'z0']]
    sub['track_id'], _ = tag_bins(sub['track_id'])
    return sub


def precision_and_recall(y_true, y_pred,threshold=0.5):
    tp,fp,fn,tn=0,0,0,0

    for i in range(0,len(y_true)):
        if y_pred[i]>=threshold:
            if y_true[i]>0:
                tp+=1
            else:
                fp+=1
        elif y_true[i]==0:
            tn+=1
        else:
            fn+=1
    precision=tp/(tp+fp) if (tp+fp != 0) else 0
    recall=tp/(tp+fn) if (tp+fn != 0) else 0
    accuracy=(tp+tn)/(tp+tn+fp+fn)
    print('Threshold',threshold,' --- Precision: {:5.4f}, Recall: {:5.4f}, Accuracy: {:5.4f}'.format(precision,recall,accuracy))
    return precision, recall, accuracy