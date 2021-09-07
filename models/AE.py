import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input, Dropout, Activation, Concatenate
from tensorflow.keras.models import Sequential, Model
from .layers import CompositePrior, NonLinear
from scipy.sparse import csr_matrix

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim), stddev=0.01)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def log_normal_diag(x, mean, log_var, axis=1, reduction=True):
    res = -0.5 * (log_var + tf.pow(x - mean, 2) / tf.exp(log_var ))
    if reduction:
        return tf.reduce_sum(res, axis=axis)
    else:
        return res

class DiagonalToZero(tf.keras.constraints.Constraint):
    def __call__(self, w):
        """Set diagonal to zero to avoid identity function"""
        q = tf.linalg.set_diag(w, tf.zeros(w.shape[0:-1]), name=None)
        return q

class BaseAE(tf.keras.models.Model):
    def __init__(self, num_items, emb_dim, hidden_layers, activation=None):
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.activation = activation
        self.hidden_layers = hidden_layers

    def compile(self, optimizer='adam', loss=None):
        super().compile()
        self.optimizer = optimizer
        self.loss = loss

    def build_encoder(self, dropout_rate=0.5):
        inputs = Input(shape = (self.num_items, ))
        
        encoder = Sequential()
        if dropout_rate > 0:
            encoder.add(Dropout(dropout_rate))

        for h in self.hidden_layers:
            encoder.add(Dense(h, activation='relu'))
            encoder.add(tf.keras.layers.LayerNormalization()) # key factor!

        encoder.add(Dense(self.emb_dim, activation=self.activation))
        
        outputs = encoder(inputs)
        
        return Model(inputs, outputs)
    
    def build_decoder(self):
        inputs = Input(shape = (self.emb_dim, ))
        
        decoder = Sequential()
        decoder.add(Dense(self.num_items))
        
        outputs = decoder(inputs)
        
        return Model(inputs, outputs)

class BaseVAE(BaseAE):
    def build_encoder(self, dropout_rate=0.5):
        inputs = Input(shape = (self.num_items, ))
        h = inputs

        if dropout_rate > 0:
            h = Dropout(dropout_rate)(h)
        
        for hidden in self.hidden_layers:
            h = Dense(hidden, activation='relu')(h)
        
        mu = Dense(self.emb_dim)(h)
        log_var = Dense(self.emb_dim)(h)
        
        return Model(inputs, [mu, log_var])

class DAE(BaseAE):
    def __init__(self, num_items, emb_dim, hidden_layers, activation=None):
        super().__init__(num_items, emb_dim, hidden_layers, activation)

        self.model = self.build_model()

    def call(self, x):
        return self.model(x)

    def train_step(self, x, y=None):
        with tf.GradientTape() as tape:
            pred = self(x)

            loss = self.loss(x, pred)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return {'loss': loss}

    def build_model(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        inputs = self.encoder.input
        outputs = self.decoder(self.encoder(inputs))
        
        return Model(inputs, outputs)
    

class CDAE(BaseAE):
    def __init__(self, num_users, num_items, emb_dim, hidden_layers, activation=None):
        super().__init__(num_items, emb_dim, hidden_layers, activation)
        self.num_users = num_users
        self.embedding = Embedding(num_users, emb_dim, )
        self.model = self.build_model()

    def call(self, data):
        user_ids, items = data
        return self.model([user_ids, items])

    def train_step(self, data):
        (user_ids, items) = data[0]

        with tf.GradientTape() as tape:
            pred = self([user_ids, items])

            loss = self.loss(items, pred)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return {'loss': loss}
        
    def build_model(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        rating = Input(shape=(self.num_items, ), name='rating_input')
        user_id = Input(shape=(1, ), name='user_input')
        
        emb = self.embedding(user_id)
        emb = Flatten()(emb)
        enc = self.encoder(rating) + emb
        enc = Activation(self.activation)(enc)
        outputs = self.decoder(enc)
    
        return Model([user_id, rating], outputs)
    

class MultVAE(BaseVAE):
    def __init__(self, num_items, emb_dim, hidden_layers, activation=None):
        super().__init__(num_items, emb_dim, hidden_layers, activation)
        self.anneal = 0.
        
        self.model = self.build_model()

    def build_model(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        inputs = self.encoder.input
        
        mu, log_var = self.encoder(inputs)
        h = sampling([mu, log_var])
        
        outputs = self.decoder(h)
    
        return Model(inputs, [outputs, mu, log_var])

    def call(self, x, training=False):
        if training:
            return self.model(x)
        else:
            pred, _, _ = self.model(x)
            return pred
    
    def predict(self, data, *args, **kwargs):
        mu, _ = self.encoder.predict(data, *args, **kwargs)
        return self.decoder.predict(mu, *args, **kwargs)

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            pred, mu, log_var = self(x, training=True)

            kl_loss = tf.reduce_sum(0.5*(-log_var + tf.exp(log_var) + tf.pow(mu, 2)-1), 1)
            ce_loss = -tf.reduce_sum(tf.nn.log_softmax(pred) * x, -1)
            loss = ce_loss + kl_loss*self.anneal
            
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {'loss': loss}


class EASE:
    def __init__(self, users, items):
        self.users = users
        self.items = items

    def fit(self, lambda_ = 1.5):
        users, items = self.users, self.items
        values = np.ones(users.shape[0])

        X = csr_matrix((values, (users, items)))
        self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)

    def predict_one_user(self, user, items):
        return np.take(self.pred[user, :], items)
        

class NeuralEASE(tf.keras.models.Model):
    def __init__(self, num_items):
        super().__init__()
        self.ease = Dense(num_items, use_bias=False, kernel_constraint=DiagonalToZero(), activation='sigmoid')

    def compile(self, optimizer, loss):
        super().compile()
        self.optimizer = optimizer
        self.loss = loss

    def call(self, x):
        return self.ease(x)

    def train_step(self, x):
        with tf.GradientTape() as tape:
            pred = self(x)
            loss = self.loss(x, pred)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {'loss': loss}
        

class HVampVAE(tf.keras.models.Model):
    def __init__(self, num_items, emb_dim, hidden_layers, gated=True, activation='tanh', dropout_rate=0.5, number_components=1000):
        super().__init__()
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.hidden_layers = hidden_layers
        self.gated = gated
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        self.q_z2 = self.build_q_z2()
        self.q_z1 = self.build_q_z1()
        self.p_z1 = self.build_p_z1()
        self.p_x = self.build_p_x()
        
        self.number_components = number_components
        self.idle_input = tf.Variable(tf.eye(number_components))
        self.means = Dense(num_items, use_bias=False, activation=lambda x: tf.clip_by_value(x, -1, 1))
        
    def compile(self, optimizer, loss=None):
        super().compile()
        self.optimizer = optimizer
        
    def call(self, x):
        z2_q_mean, z2_q_logvar = self.q_z2(x)
        z2_q = sampling([z2_q_mean, z2_q_logvar])
        
        z1_q_mean, z1_q_logvar = self.q_z1([x, z2_q])
        z1_q = sampling([z1_q_mean, z1_q_logvar])
        
        z1_p_mean, z1_p_logvar = self.p_z1(z2_q)

        pred = self.p_x([z1_q, z2_q])
        
        return pred, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar
    
    def predict(self, x, *args, **kwargs): # not implemented other options yet
        pred = self(x)[0]
        return pred.numpy()

    def log_p_z2(self, z2):
        x = self.means(self.idle_input)
        z2_p_mean, z2_p_logvar = self.q_z2(x)
        
        z2_expand = tf.expand_dims(z2, 1)
        z2_p_mean = tf.expand_dims(z2_p_mean, 0)
        z2_p_logvar = tf.expand_dims(z2_p_logvar, 0)

        a = log_normal_diag(z2_expand, z2_p_mean, z2_p_logvar, axis=2) - tf.math.log(float(self.number_components))
        a_max = tf.reduce_max(a, 1)
        log_prior = (a_max + tf.math.log(tf.reduce_sum(tf.exp(a - tf.expand_dims(a_max, 1)), 1)))

        return log_prior
        
    def train_step(self, data, beta=1.):
        x = data
        with tf.GradientTape() as tape:
            pred, z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = self(x)
            # recon loss
            # RE = tf.reduce_mean(tf.reduce_sum(tf.nn.log_softmax(pred) * x, -1))
            RE = tf.reduce_sum(tf.nn.log_softmax(pred) * x, -1)
            
            # kl loss
            log_p_z1 = log_normal_diag(z1_q, z1_p_mean, z1_p_logvar, axis=1)
            log_q_z1 = log_normal_diag(z1_q, z1_q_mean, z1_q_logvar, axis=1)
            log_p_z2 = self.log_p_z2(z2_q)
            log_q_z2 = log_normal_diag(z2_q, z2_q_mean, z2_q_logvar, axis=1)
            KL = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)    
            loss = -RE + beta * KL
            loss = tf.reduce_mean(loss)
            
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {'loss': loss, 'RE': tf.reduce_mean(RE), 'KL': tf.reduce_mean(KL)}
    
    def build_q_z2(self):
        x_in = Input(shape=(self.num_items, ))
        
        x = Dropout(self.dropout_rate)(x_in)

        for hidden in self.hidden_layers:
            x = NonLinear(hidden, gated=self.gated, activation=self.activation)(x)
            x = tf.keras.layers.LayerNormalization()(x)
        
        mu = Dense(self.emb_dim)(x)
        logvar = Dense(self.emb_dim)(x)
        
        return Model(x_in, [mu, logvar], name='q_z2')
    
    def build_q_z1(self):
        x_in = Input(shape=(self.num_items, ))
        z2_in = Input(shape=(self.emb_dim, ))
        z2 = z2_in
        
        x = Dropout(self.dropout_rate)(x_in)

        h = Concatenate()([x, z2]) 
        for hidden in self.hidden_layers:
            h = NonLinear(hidden, gated=self.gated, activation=self.activation)(h)
            h = tf.keras.layers.LayerNormalization()(h)

        mu = Dense(self.emb_dim)(h)
        logvar = Dense(self.emb_dim)(h)
        
        return Model([x_in, z2_in], [mu, logvar], name='q_z1')
    
    def build_p_z1(self):
        z2_in = Input(shape=(self.emb_dim, ))
        h = z2_in
        for hidden in self.hidden_layers:
            h = NonLinear(hidden, gated=self.gated, activation=self.activation)(h)
            h = tf.keras.layers.LayerNormalization()(h)
        
        mu = Dense(self.emb_dim)(h)
        logvar = Dense(self.emb_dim)(h)
        
        return Model(z2_in, [mu, logvar], name='p_z1')
         
    def build_p_x(self):
        z1_in = Input(shape=(self.emb_dim, ))
        z2_in = Input(shape=(self.emb_dim, ))
        z1 = z1_in
        z2 = z2_in

        h = Concatenate()([z1, z2])
        for hidden in self.hidden_layers:
            h = NonLinear(hidden, gated=self.gated, activation=self.activation)(h)
            h = tf.keras.layers.LayerNormalization()(h)

        out = Dense(self.num_items)(h)
        
        return Model([z1_in, z2_in], out, name='p_x')


class VASP(BaseVAE):
    # TODO: residual connection modulization // 
    # split augmentation is not implemented
    def __init__(self, num_items, emb_dim, hidden_layers, activation=None, sampling_ratio=0.3):
        super().__init__(num_items, emb_dim, hidden_layers, activation)
        
        self.model = self.build_model()
        self.ease = Dense(self.num_items, activation='sigmoid', use_bias=False, kernel_constraint=DiagonalToZero())
        
    def build_model(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        inputs = self.encoder.input
        
        mu, log_var = self.encoder(inputs)
        h = sampling([mu, log_var])
        
        outputs = self.decoder(h)

        return Model(inputs, [outputs, mu, log_var])

    def call(self, x, training=False):
        sampled_x = x
        
        d, mu, log_var = self.model(sampled_x)

        ease = self.ease(sampled_x)
        
        pred = d * ease # logical and
        
        if training:
            return pred, mu, log_var
        else:
            return pred
    
    def predict(self, data, *args, **kwargs):
        x = data
        sampled_x = x
        
        mu, _ = self.encoder.predict(sampled_x, *args, **kwargs)
        d = self.decoder.predict(mu, *args, **kwargs)

        ease = self.ease(sampled_x).numpy()

        pred = d * ease
        
        return pred

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            pred, mu, log_var = self(x, training=True)

            kl_loss = tf.reduce_mean(0.5*(-log_var + tf.exp(log_var) + tf.pow(mu, 2)-1), 1)
            ce_loss = self.loss(x, pred)
            loss = ce_loss + kl_loss
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {'loss': loss}
    
    def build_encoder(self):
        hidden = 600
        inputs = Input(shape = (self.num_items, ))
        h = inputs
        h = Dropout(0.5)(h)
        
        h1 = Dense(self.hidden_layers[0], activation='relu')(h)
        h1 = tf.keras.layers.LayerNormalization()(h1)

        h2 = Dense(hidden)(h1) + h1
        h2 = Activation('relu')(h2)
        h2 = tf.keras.layers.LayerNormalization()(h2)
        
        h3 = Dense(hidden)(h2) + h1 +  h2
        h3 = Activation('relu')(h3)
        h3 = tf.keras.layers.LayerNormalization()(h3)

        mu = Dense(self.emb_dim)(h3)
        log_var = Dense(self.emb_dim)(h3)
        
        return Model(inputs, [mu, log_var])
    
    def build_decoder(self):
        inputs = Input(shape = (self.emb_dim, ))
        h = inputs
        
        h0 = Dense(600, activation='relu')(h)
        h0 = tf.keras.layers.LayerNormalization()(h0)
        
        h1 = Dense(600)(h0)+h0
        h1 = Activation('relu')(h1)
        h1 = tf.keras.layers.LayerNormalization()(h1)
        
        decoder_r = Dense(self.num_items, activation='sigmoid')(h1)
        decoder_l = Dense(self.num_items, activation='sigmoid')(h)
        
        outputs = decoder_r * decoder_l # deep & wide
        
        return Model(inputs, outputs)

class RecVAE(BaseVAE):
    ## need to be fixed => seperate all models in seperate file
    def __init__(self, num_items, emb_dim, hidden_layers, activation=None, gamma=1., alternate=True):
        super().__init__(num_items, emb_dim, hidden_layers, activation)

        self.gamma = gamma
        self.alternate = alternate

        self.build_model()

    def compile(self, enc_optim, dec_optim, optimizer=tf.optimizers.Adam(), loss=None):
        super().compile(optimizer=optimizer, loss=loss)
        self.enc_optim = enc_optim
        self.dec_optim = dec_optim

    def build_model(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.prior = CompositePrior(self.emb_dim)
        self.prior.encoder_old = self.build_encoder(dropout_rate=0)
        self.prior.encoder_old.set_weights(self.encoder.get_weights())

    def call(self, data, training=False):
        mu, logvar = self.encoder(data)
        z = sampling([mu, logvar])
        recon = self.decoder(z)

        if training:
            return mu, logvar, z, recon
        else:
            return recon
    
    def predict(self, data):
        mu, logvar = self.encoder(data)
        z = sampling([mu, logvar])
        recon = self.decoder(z)
        
        return recon
    
    def get_loss(self, x):
        norm = tf.reduce_sum(x, -1, keepdims=True)
        kl_weight = self.gamma*norm
        mu, logvar, z, pred = self(x, training=True)

        kl_loss = tf.reduce_mean(log_normal_diag(z, mu, logvar, reduction=False) - tf.multiply(self.prior(x, z), kl_weight))
        ce_loss = -tf.reduce_mean(tf.reduce_sum(tf.nn.log_softmax(pred) * x, -1))
        
        loss = ce_loss + kl_loss*kl_weight

        return loss

    def train_step(self, x, y=None):
        if self.alternate:
            # encoder
            with tf.GradientTape() as tape:
                enc_loss = self.get_loss(x)

            grads = tape.gradient(enc_loss, self.encoder.trainable_weights)
            self.enc_optim.apply_gradients(zip(grads, self.encoder.trainable_weights))

            # self.update_prior() => should be out of model class
            # decoder
            with tf.GradientTape() as tape:
                dec_loss = self.get_loss(x)

            grads = tape.gradient(dec_loss, self.decoder.trainable_weights)
            self.dec_optim.apply_gradients(zip(grads, self.decoder.trainable_weights))

            return {'encoder_loss': enc_loss, 'decoder_loss': dec_loss}

        else:
            with tf.GradientTape() as tape:
                loss = self.get_loss(x)
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            return {'loss': loss}

    def update_prior(self):
        self.prior.encoder_old.set_weights(self.encoder.get_weights())