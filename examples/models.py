from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, layers, losses
from tensorflow.keras.layers import Dense, Concatenate, Activation, Add, BatchNormalization, Dropout, Input, Embedding, Flatten, Multiply
from tensorflow.keras.models import Model, Sequential, load_model

# NMF
class GMF(keras.Model):
    def __init__(self, u_dim, i_dim, latent_dim):
        super(GMF, self).__init__()
        
        self.u_dim = u_dim
        self.i_dim = i_dim
        self.latent_dim = latent_dim
        
        self.model = self.build_model()

    def compile(self, optim, loss_fn):
        super(GMF, self).compile()
        self.optim = optim
        self.loss_fn = loss_fn
    
    def build_model(self):
        u_input = Input(shape=(1, ))
        i_input = Input(shape=(1, ))

        u_emb = Flatten()(Embedding(self.u_dim, self.latent_dim, input_length=u_input.shape[1])(u_input))
        i_emb = Flatten()(Embedding(self.i_dim, self.latent_dim, input_length=i_input.shape[1])(i_input))

        mul = Multiply()([u_emb, i_emb])

        out = Dense(1)(mul)
        
        return Model([u_input, i_input], out)
    
    def train_step(self, data):
        user, item, y = data

        with tf.GradientTape() as tape:
            pred = self.model([user, item])
            loss = self.loss_fn(y, pred)
            
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return {'loss': loss}
    
    def call(self, data):
        user, item = data
        return self.model([user, item])


class MLP(keras.Model):
    def __init__(self, u_dim, i_dim, latent_dim):
        super(MLP, self).__init__()
        
        self.u_dim = u_dim
        self.i_dim = i_dim
        self.latent_dim = latent_dim
        
        self.model = self.build_model()

    def compile(self, optim, loss_fn):
        super(MLP, self).compile()
        self.optim = optim
        self.loss_fn = loss_fn
    
    def build_model(self):
        u_input = Input(shape=(1, ))
        i_input = Input(shape=(1, ))

        u_emb = Flatten()(Embedding(self.u_dim, self.latent_dim, input_length=u_input.shape[1])(u_input))
        i_emb = Flatten()(Embedding(self.i_dim, self.latent_dim, input_length=i_input.shape[1])(i_input))

        concat = Concatenate()([u_emb, i_emb])
        
        h = Dense(128, activation='relu')(concat)
        h = Dropout(0.2)(h)
        h = Dense(64, activation='relu')(h)
        h = Dropout(0.2)(h)

        out = Dense(1)(h)
        
        return Model([u_input, i_input], out)
    
    def train_step(self, data):
        user, item, y = data

        with tf.GradientTape() as tape:
            pred = self.model([user, item])
            loss = self.loss_fn(y, pred)
            
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return {'loss': loss}
    
    def call(self, data):
        user, item = data
        return self.model([user, item])

class MLP(keras.Model):
    def __init__(self, u_dim, i_dim, latent_dim):
        super(MLP, self).__init__()
        
        self.u_dim = u_dim
        self.i_dim = i_dim
        self.latent_dim = latent_dim
        
        self.model = self.build_model()

    def compile(self, optim, loss_fn):
        super(MLP, self).compile()
        self.optim = optim
        self.loss_fn = loss_fn
    
    def build_model(self):
        u_input = Input(shape=(1, ))
        i_input = Input(shape=(1, ))

        u_emb = Flatten()(Embedding(self.u_dim, self.latent_dim, input_length=u_input.shape[1])(u_input))
        i_emb = Flatten()(Embedding(self.i_dim, self.latent_dim, input_length=i_input.shape[1])(i_input))

        concat = Concatenate()([u_emb, i_emb])
        
        h = Dense(64, activation='relu')(concat)
        h = Dropout(0.2)(h)
        h = Dense(32, activation='relu')(h)
        h = Dropout(0.2)(h)

        out = Dense(1)(h)
        
        return Model([u_input, i_input], out)
    
    def train_step(self, data):
        user, item, y = data

        with tf.GradientTape() as tape:
            pred = self.model([user, item])
            loss = self.loss_fn(y, pred)
            
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return {'loss': loss}
    
    def call(self, data):
        user, item = data
        return self.model([user, item])
    
class NMF(keras.Model):
    def __init__(self, u_dim, i_dim, gmf_dim, mlp_dim):
        super(NMF, self).__init__()
        
        self.u_dim = u_dim
        self.i_dim = i_dim
        self.gmf_dim = gmf_dim
        self.mlp_dim = mlp_dim
        
        
        self.model = self.build_model()

    def compile(self, optim, loss_fn):
        super(NMF, self).compile()
        self.optim = optim
        self.loss_fn = loss_fn
    
    def build_model(self):
        u_input = Input(shape=(1, ))
        i_input = Input(shape=(1, ))

        u_emb_gmf = Flatten()(Embedding(self.u_dim, self.gmf_dim, input_length=u_input.shape[1])(u_input))
        i_emb_gmf = Flatten()(Embedding(self.i_dim, self.gmf_dim, input_length=i_input.shape[1])(i_input))
    
        u_emb_mlp = Flatten()(Embedding(self.u_dim, self.mlp_dim, input_length=u_input.shape[1])(u_input))
        i_emb_mlp = Flatten()(Embedding(self.i_dim, self.mlp_dim, input_length=i_input.shape[1])(i_input))
        
        # gmf
        mul = Multiply()([u_emb_gmf, i_emb_gmf])

        # mlp
        concat = Concatenate()([u_emb_mlp, i_emb_mlp])
        
        h = Dense(64, activation='relu')(concat)
        h = Dropout(0.2)(h)
        h = Dense(32, activation='relu')(h)
        h = Dropout(0.2)(h)

        con = Concatenate()([mul, h])
        
        out = Dense(1)(con)
        
        return Model([u_input, i_input], out)
    
    def train_step(self, data):
        user, item, y = data

        with tf.GradientTape() as tape:
            pred = self.model([user, item])
            loss = self.loss_fn(y, pred)
            
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return {'loss': loss}
    
    def call(self, data):
        user, item = data
        return self.model([user, item])

# Wide & Deep
class WideAndDeep(keras.Model):
    def __init__(self, u_dim, i_dim, u_emb_dim=4, i_emb_dim=4):
        super(WideAndDeep, self).__init__()
        
        self.u_dim = u_dim
        self.i_dim = i_dim
        self.u_emb_dim = u_emb_dim
        self.i_emb_dim = i_emb_dim
        
        self.deep_model = self.build_deep_model()
        self.wide_model = self.build_wide_model()


    def compile(self, wide_optim, deep_optim, loss_fn):
        super(WideAndDeep, self).compile()
        self.wide_optim = wide_optim
        self.deep_optim = deep_optim
        self.loss_fn = loss_fn
    
    def build_deep_model(self):
        u_input = Input(shape=(1, ))
        i_input = Input(shape=(1, ))

        u_emb = Flatten()(Embedding(self.u_dim, self.u_emb_dim, input_length=u_input.shape[1])(u_input))
        i_emb = Flatten()(Embedding(self.i_dim, self.i_emb_dim, input_length=i_input.shape[1])(i_input))

        concat = Concatenate()([u_emb, i_emb])
        
        h = Dense(128, activation='relu')(concat)
        h = Dropout(0.2)(h)
        h = Dense(256, activation='relu')(h)
        h = Dropout(0.2)(h)
        h = Dense(32, activation='relu')(h)
        h = Dropout(0.2)(h)

        out = Dense(1)(h)
        
        return Model([u_input, i_input], out, name='DeepModel')
    
    def build_wide_model(self):
        u_input = Input(shape=(self.u_dim, ))
        i_input = Input(shape=(self.i_dim, ))

        concat = Concatenate()([u_input, i_input])
        
        out = Dense(1)(concat)
        
        return Model([u_input, i_input], out, name='WideModel')
        
    
    def train_step(self, data):
        X, y = data
        user, item, user_ohe, item_ohe = X
        
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            wide_logit = self.wide_model([user_ohe, item_ohe])
            deep_logit = self.deep_model([user, item])
            logit = 0.5*(wide_logit + deep_logit)
            
            loss = self.loss_fn(y, logit)
            
        wide_grads = tape1.gradient(loss, self.wide_model.trainable_weights)
        self.wide_optim.apply_gradients(zip(wide_grads, self.wide_model.trainable_weights))
        
        deep_grads = tape2.gradient(loss, self.deep_model.trainable_weights)
        self.deep_optim.apply_gradients(zip(deep_grads, self.deep_model.trainable_weights))
        
        return {'loss': loss}
    
    def call(self, data):
        user, item, user_ohe, item_ohe = data
        wide_logit = self.wide_model([user_ohe, item_ohe])
        deep_logit = self.deep_model([user, item])
        return 0.5*(wide_logit + deep_logit)
    
class FM_layer(keras.Model):
    def __init__(self, x_dim, latent_dim, w_reg=1e-4, v_reg=1e-4):
        super(FM_layer, self).__init__()
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w_0 = self.add_weight(shape=(1, ),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        
        self.w = self.add_weight(shape=(self.x_dim, 1), 
                             initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=l2(self.w_reg))
        
        self.V = self.add_weight(shape=(self.x_dim, self.latent_dim), 
                             initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=l2(self.v_reg))

    def call(self, inputs):
        linear_terms = tf.reduce_sum(tf.matmul(inputs, self.w), axis=1)

        interactions = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(inputs, self.V), 2)
            - tf.matmul(tf.pow(inputs, 2), tf.pow(self.V, 2)),
            1,
            keepdims=False
        )

        y_hat = (self.w_0 + linear_terms + interactions)

        return y_hat

