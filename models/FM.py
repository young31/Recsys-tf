import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from .layers import *

class AFM(tf.keras.models.Model):
    def __init__(self, x_dims, emb_dim, att_dim):
        '''
        x_dims = [num_users, num_items, ...]
        it is for general purpose(extra features)
        '''
        super().__init__()
        self.x_dims = x_dims
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        
        self.embedding = Embedding(sum(x_dims)+1, emb_dim)
        
        self.linear = Dense(1)
        self.attention = Dense(att_dim)
        self.projection = Dense(1)
        self.attention_out = Dense(1)

    def call(self, inputs):
        '''
        inputs = [user, item, ...]
        '''
        cat_ = [tf.squeeze(tf.one_hot(feat, self.x_dims[i]), 1) for i, feat in enumerate(inputs)]
        X_cat = tf.concat(cat_, 1)
        linear_out = self.linear(X_cat)
        X = tf.concat(inputs, 1)
        non_zero_emb = self.embedding(X + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])

        n = len(self.x_dims)
        r = []; c = []
        for i in range(n-1):
            for j in range(i+1, n):
                r.append(i), c.append(j)
        p = tf.gather(non_zero_emb, r, axis=1)
        q = tf.gather(non_zero_emb, c, axis=1)
        pairwise = p*q
        
        att_score = tf.nn.relu(self.attention(pairwise))
        att_score = tf.nn.softmax(self.projection(att_score), axis=1)

        att_out = tf.reduce_sum(att_score * pairwise, axis=1)

        att_out = self.attention_out(att_out)
        
        print(linear_out, att_out)
        out = att_out + linear_out

        return out


class AutoInt(tf.keras.models.Model):
    def __init__(self, x_dims, emb_dim, att_sizes, att_heads):
        super().__init__()
        self.x_dims = x_dims
        self.emb_dim = emb_dim

        self.embedding = Embedding(sum(x_dims)+1, emb_dim)
        
        self.final_out = Dense(1)
        
        self.att_layers = [MHA(a, h) for a, h in zip(att_sizes, att_heads)]
        
        self.flatten =  Flatten()
        
    def call(self, inputs):
        inputs = tf.concat(inputs, 1)
        emb = self.embedding(inputs + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])
        
        att = emb
        for att_layer in self.att_layers:
            att = att_layer(att)
        
        out = self.final_out(self.flatten(att))
        
        return out


class CDN(tf.keras.models.Model):
    def __init__(self, x_dims, emb_dim, n_cross_layers, hidden_layers, activation='relu'):
        super().__init__()
        self.x_dims = x_dims
        self.emb_dim = emb_dim
        
        self.cross_layers = CrossNetwork(n_cross_layers)
        self.deep_layers = DeepNetwork(hidden_layers, activation)
        
        self.embedding = Embedding(sum(x_dims)+1, emb_dim)
    
        self.flatten = Flatten()
        self.final_out = Dense(1)
        
    def call(self, inputs):
        inputs = tf.concat(inputs, 1)
        embed = self.embedding(inputs + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])
        embed = self.flatten(embed)
        
        # if continuous, concat with embed
        cross_out = self.cross_layers(embed)
        deep_out = self.deep_layers(embed)
        
        out = tf.concat([cross_out, deep_out], 1)

        out = self.final_out(out)
        
        return out


class DeepFM(tf.keras.models.Model):
    def __init__(self, x_dims, emb_dim, hidden_layers):
        super().__init__()
        self.x_dims = x_dims
        self.emb_dim = emb_dim

        self.embedding = Embedding(sum(x_dims)+1, emb_dim)
        self.fm_layer = FM(emb_dim)
        self.hidden_layers = hidden_layers
        self.dnn_layers = self.build_dnn()
        self.flatten =  Flatten()

    def build_dnn(self):
        model = Sequential()
        for h in self.hidden_layers:
            model.add(Dense(h, activation='relu'))
        model.add(Dense(1))

        return model
        
    def call(self, inputs):      
        inputs = tf.concat(inputs, 1)  
        emb = self.embedding(inputs + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])
        embed = self.flatten(emb)

        fm_out = self.fm_layer(embed)

        deep_out = self.dnn_layers(embed)

        out = fm_out + deep_out

        return out


class PNN(tf.keras.models.Model):
    def __init__(self, x_dims, emb_dim, hidden_layers, model_type='inner'):
        super().__init__()
        self.x_dims = x_dims
        self.emb_dim = emb_dim

        self.embedding = Embedding(sum(x_dims)+1, emb_dim)

        self.linear = Dense(emb_dim)

        if model_type == 'inner':
            self.pnn = InnerProduct(x_dims)
        elif model_type == 'outer':
            self.pnn = OuterProduct(x_dims)
        else:
            raise ValueError('no available model type')
        
        self.dnn = [Dense(unit, activation='relu') for unit in hidden_layers]
        
        self.final_out = Dense(1)
        
        self.flatten = Flatten()
        
    def call(self, inputs):
        inputs = tf.concat(inputs, 1)
        emb = self.embedding(inputs + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])
        
        linear = self.flatten(self.linear(emb))
        quadratic = self.pnn(emb)

        concat = tf.concat([linear, quadratic], -1)
        
        out = concat
        for layer in self.dnn:
            out = layer(out)
        
        out = self.final_out(out)
        return out  


class xDFM(tf.keras.models.Model):
    def __init__(self, x_dims, emb_dim, cin_layers, hidden_layers, activation='relu'):
        super().__init__()
        self.x_dims = x_dims
        
        self.embedding = Embedding(sum(x_dims)+1, emb_dim)
        
        self.linear = Dense(1)
        
        self.dnn_layers = [Dense(n, activation=activation) for n in hidden_layers]
        self.dnn_final = Dense(1)
        
        self.cin_layers = CIN(cin_layers, activation=activation)
        self.cin_final = Dense(1)
        
    def call(self, inputs):
        # only apply ohe for categorical
        sparse = [tf.squeeze(tf.one_hot(feat, self.x_dims[i]), 1) for i, feat in enumerate(inputs)]
        sparse = tf.concat(sparse, 1)
        linear_out = self.linear(sparse)

        x = tf.concat(inputs, 1)
        emb = self.embedding(x + tf.constant((0, *np.cumsum(self.x_dims)))[:-1])
        
        dnn_input = Flatten()(emb)
        dnn_out = dnn_input
        for dnn_layer in self.dnn_layers:
            dnn_out = dnn_layer(dnn_out)
        dnn_out = self.dnn_final(dnn_out)

        cin_out = self.cin_layers(emb)
        cin_out = self.cin_final(cin_out)

        out = linear_out + dnn_out + cin_out
        
        return out