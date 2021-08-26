import numpy as np
import heapq

def evaluate_model(model, testRatings, testNegatives, K=10):
    hits, ndcgs = [],[]
    for idx in range(len(testRatings)):
        user = testRatings[idx][0]
        gtItem = testRatings[idx][1]
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