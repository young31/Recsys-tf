import numpy as np
import pandas as pd
import argparse

from models.MF import *
from models.FM import *
from data_load import *
from evaluate import *

def parse_args():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--resample', action='store_true')
    # data
    parser.add_argument('--n_negatives', type=int, default=4)
    # model
    parser.add_argument('--model_name', type=str, default='NMF')
    parser.add_argument('--emb_dim', type=int, default=16)
    parser.add_argument('--hidden_layers', default='[128, 64]') # 늘려서 해보기
    parser.add_argument('--att_dim', type=int, default=8)
    parser.add_argument('--att_sizes', default='[16, 16]')
    parser.add_argument('--att_heads', default='[4, 4]')
    parser.add_argument('--n_cross_layers', type=int, default=2)
    parser.add_argument('--cin_layers', default='[32, 32]')
    parser.add_argument('--model_type', type=str, default='inner')
    parser.add_argument('--activation', type=str, default='relu')
    return parser.parse_args()

def get_model(args):
    model_name = args.model_name
    if model_name == 'GMF':
        return GMF(args.num_users, args.num_items, args.emb_dim)
    elif model_name == 'MLP':
        return MLP(args.num_users, args.num_items, args.emb_dim, args.hidden_layers)
    elif model_name == 'NMF':
        return NMF(args.num_users, args.num_items, args.emb_dim, args.hidden_layers)
    elif model_name == 'AFM':
        return AFM(args.x_dims, args.emb_dim, args.att_dim)
    elif model_name == 'AutoInt':
        return AutoInt(args.x_dims, args.emb_dim, args.att_sizes, args.att_heads)
    elif model_name == 'CDN':
        return CDN(args.x_dims, args.emb_dim, args.n_cross_layers, args.hidden_layers, args.activation)
    elif model_name == 'DeepFM':
        return DeepFM(args.x_dims, args.emb_dim, args.hidden_layers)
    elif model_name == 'PNN':
        return PNN(args.x_dims, args.emb_dim, args.hidden_layers, args.model_type)
    elif model_name == 'xDFM':
        return xDFM(args.x_dims, args.emb_dim, args.cin_layers, args.hidden_layers, args.activation)
    else:
        raise(ValueError('not available model'))


if __name__=='__main__':
    train = load_rating_file_as_matrix('./data/ml-1m.train.rating')
    test_ratings = load_rating_file_as_list('./data/ml-1m.test.rating')
    test_negatives = load_negative_file('./data/ml-1m.test.negative')

    num_users, num_items = train.shape

    args = parse_args()
    print(args)
    args.hidden_layers = eval(args.hidden_layers)
    args.cin_layers = eval(args.cin_layers)
    args.att_sizes = eval(args.att_sizes)
    args.att_heads = eval(args.att_heads)

    args.num_users = num_users
    args.num_items = num_items
    args.x_dims = [num_users, num_items]

    # data
    user_inputs, item_inputs, labels = get_train_instances(train, args.n_negatives)
    # model
    model = get_model(args)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.BinaryCrossentropy(from_logits=True))
    # eval before
    hr_hist = []
    ndcg_hist = []
    hr, ndcg = evaluate_model(model, test_ratings, test_negatives)
    hr_hist.append(np.mean(hr))
    ndcg_hist.append(np.mean(ndcg))
    # train
    for _ in range(args.epochs):
        model.fit(
            [np.array(user_inputs), np.array(item_inputs)], np.array(labels),
            batch_size = args.batch_size,
            shuffle=True,
        )

        hr, ndcg = evaluate_model(model, test_ratings, test_negatives)
        hr_hist.append(np.mean(hr))
        ndcg_hist.append(np.mean(ndcg))

        if args.resample:
            user_inputs, item_inputs, labels = get_train_instances(train, args.n_negatives)

    # save
    model.save_weights(f'./weights/{args.model_name}.h5')
    res = pd.DataFrame(index=range(len(hr_hist)), columns=['hr', 'ndcg'])
    res['hr'] = hr_hist
    res['ndcg'] = ndcg_hist

    res.to_csv(f'./scores/{args.model_name}.csv', index=False)