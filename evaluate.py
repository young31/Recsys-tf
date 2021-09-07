import numpy as np
from models.AE import CDAE
import heapq

def evaluate_model(model, testRatings, testNegatives, history=None, K=10, is_ae=False, is_ease=False):
    hits, ndcgs = [],[]
    for idx in range(len(testRatings)):
        user = testRatings[idx][0]
        gtItem = testRatings[idx][1]
        if is_ae:
            user = testRatings[idx][0] if isinstance(model, CDAE) else None
            (hr,ndcg) = eval_one_rating_ae(model, gtItem, testNegatives[idx], history[idx:idx+1], user=user, K=K)
        elif is_ease:
            (hr,ndcg) = eval_one_rating_ease(model, gtItem, testNegatives[idx], user=user, K=K)
        else:
            (hr,ndcg) = eval_one_rating(model, user, gtItem, testNegatives[idx], K)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(model, user, gtItem, negatives, K):
    items = negatives + [gtItem]
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), user, dtype = 'int32')
    predictions = model.predict([users, np.array(items)], 
                                batch_size=100, verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def eval_one_rating_ae(model, gtItem, negatives, history, user=None, K=10):
    items = negatives + [gtItem]
    # Get prediction scores
    map_item_score = {}
    if user is not None:
        users = users = np.full((1, 1), user, dtype = 'int32')
        predictions = model.predict([users, history], 
                                    batch_size=100, verbose=0)
        predictions = predictions[0][items]
    else:
        predictions = model.predict(history, 
                                    batch_size=100, verbose=0)
        predictions = predictions[0][items]

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def eval_one_rating_ease(model, gtItem, negatives, user, K=10):
    items = negatives + [gtItem]
    # Get prediction scores
    map_item_score = {}
    predictions = model.predict_one_user(user, items)

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return np.log(2) / np.log(i+2)
    return 0