import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, Flatten, Multiply, Concatenate
from tensorflow.keras.models import Sequential


class GMF(tf.keras.models.Model):
    def __init__(self, num_users, num_items, emb_dim):
        super().__init__()
        self.u_emb = Embedding(num_users, emb_dim)
        self.i_emb = Embedding(num_items, emb_dim)
        self.out = Dense(1)

    def call(self, inputs):
        users, items = inputs
        user_emb = Flatten()(self.u_emb(users))
        item_emb = Flatten()(self.i_emb(items))

        mul = Multiply()([user_emb, item_emb])

        out = self.out(mul)
        return out

class MLP(tf.keras.models.Model):
    def __init__(self, num_users, num_items, emb_dim, hidden_layers):
        super().__init__()
        self.u_emb = Embedding(num_users, emb_dim)
        self.i_emb = Embedding(num_items, emb_dim)
        self.linear = Sequential([Dense(layer, activation='relu') for layer in hidden_layers]) # how about gated?
        self.out = Dense(1)

    def call(self, inputs):
        users, items = inputs
        user_emb = Flatten()(self.u_emb(users))
        item_emb = Flatten()(self.i_emb(items))

        h = Concatenate()([user_emb, item_emb])

        out = self.linear(h)
        out = self.out(out)

        return out


class NMF(tf.keras.models.Model):
    def __init__(self, num_users, num_items, emb_dim, hidden_layers):
        super().__init__()
        self.u_emb_gmf = Embedding(num_users, emb_dim)
        self.i_emb_gmf = Embedding(num_items, emb_dim)

        self.u_emb_mlp = Embedding(num_users, emb_dim)
        self.i_emb_mlp = Embedding(num_items, emb_dim)

        self.linear = Sequential([Dense(layer, activation='relu') for layer in hidden_layers])
        self.out = Dense(1)

    def call(self, inputs):
        users, items = inputs
        # GMF part
        user_emb_gmf = Flatten()(self.u_emb_gmf(users))
        item_emb_gmf = Flatten()(self.i_emb_gmf(items))
        gmf = Multiply()([user_emb_gmf, item_emb_gmf])
        # MLP part
        user_emb_mlp = Flatten()(self.u_emb_mlp(users))
        item_emb_mlp = Flatten()(self.i_emb_mlp(items))
        mlp = Concatenate()([user_emb_mlp, item_emb_mlp])
        mlp = self.linear(mlp)

        h = Concatenate()([gmf, mlp])

        out = self.out(h)
        
        return out
