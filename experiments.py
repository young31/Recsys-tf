import os
import time
import gc

# models = [
#     'GMF', 'MLP', 'NMF', 'AFM', 'AutoInt', 'CDN', 'DeepFM', 'PNN', 'xDFM'
# ]

# for model_name in models[::-1]:
#     print(model_name)
#     os.system(f'python train.py --model_name {model_name} --epochs 30 --resample')
#     time.sleep(60)
#     gc.collect()

models = [
    'DAE',# 'CDAE'
]

for model_name in models:
    print(model_name)
    os.system(f'python train.py --model_name {model_name} --epochs 30 --resample --hidden_layers [600,600]')
    time.sleep(1)
    gc.collect()