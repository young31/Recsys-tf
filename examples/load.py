import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data(filname, threshold=0):
    # only valid for currnet workspace
    f = open(filname, 'r')
    fs = f.readlines()
    f.close()

    df = pd.DataFrame(list(map(lambda x: x.split('\t'), fs)), columns=['userId', 'movieId', 'rating', 'time'])
    df = df.drop('time', axis=1)
    df['userId'] = df['userId'].astype(int)
    df['movieId'] = df['movieId'].astype(int)
    df['rating'] = df['rating'].astype(float)
    
    df = df[['userId', 'movieId', 'rating']]
    if threshold > 0:
        df['rating'] = np.where(df['rating']>threshold, 1, 0)  
    else:
        df['rating'] = 1.
    m_codes = df['movieId'].astype('category').cat.codes
    u_codes = df['userId'].astype('category').cat.codes
    df['movieId'] = m_codes
    df['userId'] = u_codes
    
    return df

def add_negative(df, uiid, times=4):
    df_ = df.copy()
    user_id = df_['userId'].unique()
    item_id = df_['movieId'].unique()
    
    for i in tqdm(user_id):
        cnt = 0
        n = len(df_[df_['userId']==i])
        n_negative = min(n*times, len(item_id)-n-1)
        available_negative = list(set(uiid) - set(df[df['userId']==i]['movieId'].values))
        
        new = np.random.choice(available_negative, n_negative, replace=False)
        new = [[i, j, 0] for j in new]
        df_ = df_.append(pd.DataFrame(new, columns=df.columns), ignore_index=True)
    
    return df_

def extract_from_df(df, n_positive, n_negative):
    df_ = df.copy()
    rtd = []
    
    user_id = df['userId'].unique()
    
    for i in tqdm(user_id):
        rtd += list(np.random.choice(df[df['userId']==i][df['rating']==1]['movieId'].index, n_positive, replace=False))
        rtd += list(np.random.choice(df[df['userId']==i][df['rating']==0]['movieId'].index, n_negative, replace=False))
        
    return rtd