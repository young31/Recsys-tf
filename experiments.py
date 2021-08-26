import os
import time
import gc

models = [
    'GMF', 'MLP', 'NMF', 'AFM', 'AutoInt', 'CDN', 'DeepFM', 'PNN', 'xDFM'
]

for model_name in models[::-1]:
    print(model_name)
    os.system(f'python train.py --model_name {model_name} --epochs 30 --resample')
    time.sleep(60)
    gc.collect()