import heapq    
import numpy as np
import pandas as pd

def eval_hit(model, df, test, user_id, item_ids, top_k):
    df = pd.concat([df, test])
    items = list(set(item_ids) - set(df[df['userId']==user_id][df['rating']==1]['movieId'].values))
    np.random.shuffle(items)
    items = items[:99]
    items.append(test[test['userId']==user_id]['movieId'].values[0])
    items = np.array(items).reshape(-1, 1)

    user = np.full(len(items), user_id).reshape(-1, 1)

    preds = model.predict([user, items]).flatten()
    item_to_pred = {item: pred for item, pred in zip(items.flatten(), preds)}

    top_k = heapq.nlargest(top_k, item_to_pred, key=item_to_pred.get)
    
    if items[-1][0] in top_k:
            return 1
    return 0

def eval_NDCG(model, df, test, user_id, item_ids, top_k):
    df = pd.concat([df, test])
    items = list(set(item_ids) - set(df[df['userId']==user_id][df['rating']==1]['movieId'].values))
    np.random.shuffle(items)
    items = items[:99]
    items.append(test[test['userId']==user_id]['movieId'].values[0])
    items = np.array(items).reshape(-1, 1)

    user = np.full(len(items), user_id).reshape(-1, 1)

    preds = model.predict([user, items]).flatten()
    item_to_pred = {item: pred for item, pred in zip(items.flatten(), preds)}

    top_k = heapq.nlargest(top_k, item_to_pred, key=item_to_pred.get)
    
    for i, item in enumerate(top_k, 1):
        if item == test[test['userId']==user_id]['movieId'].values:
            return 1 / np.log2(i+1)
    return 0

def eval_hit_wrapper(model, df, test, item_ids, top_k):
    def f(user_id):
        return eval_hit(model, df, test, user_id, item_ids, top_k)
    return f

def eval_NDCG_wrapper(model, df, test, item_ids, top_k):
    def f(user_id):
        return eval_NDCG(model, df, test, user_id, item_ids, top_k)
    return f