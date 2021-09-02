import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input, Dropout, Activation
from tensorflow.keras.models import Sequential, Model
from .layers import CompositePrior

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
        # x = tf.nn.l2_normalize(tf.cast(x, tf.float32), 1)
        with tf.GradientTape() as tape:
            pred, mu, log_var = self(x, training=True)

            kl_loss = tf.reduce_mean(tf.reduce_sum(0.5*(log_var + tf.exp(log_var) + tf.pow(mu, 2)-1), 1, keepdims=True))
            ce_loss = -tf.reduce_mean(tf.reduce_sum(tf.nn.log_softmax(pred) * x, -1))
            # ce_loss = tf.reduce_mean(tf.losses.categorical_crossentropy(x, pred, from_logits=True))
            loss = ce_loss + kl_loss*self.anneal
            
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {'loss': loss}
    

    


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