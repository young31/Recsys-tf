import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + tf.pow((x - mu), 2) / tf.exp(logvar))

class FM(tf.keras.models.Model):
    def __init__(self, latent_dim, w_reg=1e-4, v_reg=1e-4):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w_0 = self.add_weight(shape=(1, ),
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
        
        self.w = self.add_weight(shape=(input_shape[-1], 1), 
                                initializer=tf.random_normal_initializer(),
                                trainable=True,
                                regularizer=l2(self.w_reg))
        
        self.V = self.add_weight(shape=(input_shape[-1], self.latent_dim), 
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

        out = (self.w_0 + linear_terms + interactions)
        out = tf.reshape(out, (-1, 1))

        return out    


class MHA(tf.keras.layers.Layer):
    def __init__(self, emb_size, head_num, use_resid=True):
        super(MHA, self).__init__()
        
        self.emb_size = emb_size
        self.head_num = head_num
        self.use_resid = use_resid
        
        self.flatten = Flatten()
        # officially available MHA in tf 2.6
        self.att = tfa.layers.MultiHeadAttention(emb_size, head_num)
        
    def build(self, input_shape):
        units = self.emb_size * self.head_num
        
        self.W_q = Dense(units)
        self.W_k = Dense(units)
        self.W_v = Dense(units)
        if self.use_resid:
            self.W_res = Dense(units)
            
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        q = self.W_q(inputs)
        k = self.W_k(inputs)
        v = self.W_v(inputs)
        
        out = self.att([q, k, v])

        if self.use_resid:
            out = out + self.W_res((inputs))
            
        out = tf.nn.relu(out)
        
        return out


class CrossNetwork(tf.keras.layers.Layer):
    def __init__(self, n_layers):
        super(CrossNetwork, self).__init__()
        self.n_layers = n_layers
    
    def build(self, input_shape):
        dim = input_shape[-1]
        self.cross_weights = [self.add_weight(shape=(dim, 1), 
                                        initializer=tf.random_normal_initializer(),
                                        trainable=True,
                                        name=f'cross_weight_{i}') for i in range(self.n_layers)]
    
        self.cross_biases = [self.add_weight(shape=(dim, 1),
                                        initializer=tf.random_normal_initializer(),
                                        trainable=True,
                                        name=f'cross_bias_{i}') for i in range(self.n_layers)]
    def call(self, inputs):
        x_0 = tf.expand_dims(inputs, -1)
        x_l = x_0
        for i in range(self.n_layers):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])
            x_l = tf.matmul(x_0, x_l1) + self.cross_biases[i]
            
        x_l = tf.squeeze(x_l, -1)
        
        return x_l

class DeepNetwork(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu'):
        super(DeepNetwork, self).__init__()
        
        self.layers = [Dense(unit, activation=activation) for unit in units]
    
    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            
        return x


class InnerProduct(tf.keras.layers.Layer):
    def __init__(self, x_dims):
        super().__init__()
        self.x_dims = x_dims
        
    def call(self, inputs):
        n = len(self.x_dims)
        
        p = []
        q = []
        for i in range(n):
            for j in range(i+1, n):
                p.append(i)
                q.append(j)
                
        p = tf.gather(inputs, p, axis=1)
        q = tf.gather(inputs, q, axis=1)
        
        out = p*q
        out = tf.squeeze(out, 1)
        return out
    
    
class OuterProduct(tf.keras.layers.Layer):
    def __init__(self, x_dims, kernel_type='mat'):
        super().__init__()
        self.x_dims = x_dims
        self.kernel_type = kernel_type
        
    def build(self, input_shape):
        n, m, k = input_shape
        
        if self.kernel_type == 'mat':
            self.kernel = self.add_weight(shape=(k, (m*(m-1)//2), k), 
                                            initializer = tf.zeros_initializer())
        else:
            self.kernel = self.add_weight(shape=((m*(m-1)//2), k),
                                            initializer = tf.zeros_initializer())
        
    def call(self, inputs):
        n = len(self.x_dims)
        
        p = []
        q = []
        for i in range(n):
            for j in range(i+1, n):
                p.append(i)
                q.append(j)
                
        p = tf.gather(inputs, p, axis=1)
        q = tf.gather(inputs, q, axis=1)
        
        if self.kernel_type == 'mat':
            kp = tf.transpose(tf.reduce_sum(tf.expand_dims(p, 1) * self.kernel, -1), [0, 2, 1])
            out = tf.reduce_sum(kp * q, -1)
        else:
            out = tf.reduce_sum(p * q * tf.expand_dims(self.kernel, 0), -1)
            
        return out



class CIN(tf.keras.layers.Layer):
    def __init__(self, cross_layer_sizes, activation=None):
        super(CIN, self).__init__()
        self.cross_layer_sizes = cross_layer_sizes
        self.n_layers = len(cross_layer_sizes)
        self.activation = None
        
        if activation:
            self.activation = Activation(activation)
        
        self.cross_layers = []
        for corss_layer_size in cross_layer_sizes:
            self.cross_layers.append(Conv1D(corss_layer_size, 1, data_format='channels_first'))
            
        self.linear = Dense(1)
    
    def call(self, inputs): # embedding is input
        batch_size, field_size, emb_size = inputs.shape
        xs = [inputs]

        for i, layer in enumerate(self.cross_layers):
            x = tf.einsum('nie,nje->nije', xs[i], xs[0])
            x = tf.reshape(x, (-1, field_size*xs[i].shape[1] , emb_size))

            x = layer(x)
            if self.activation:
                x = self.activation(x)
            
            xs.append(x)
            
        res = tf.reduce_sum(tf.concat(xs, axis=1), -1)
        return res


class CompositePrior(tf.keras.models.Model):
    def __init__(self, latent_dim, mixture_weights = [3/20, 15/20, 2/20]):
        super().__init__()
        self.encoder_old = None
        self.latent_dim = latent_dim
        self.mixture_weights = mixture_weights
        
        self.mu_prior = self.add_weight(shape=(self.latent_dim, ), initializer = tf.zeros_initializer(), trainable=False)
        self.logvar_prior  = self.add_weight(shape=(self.latent_dim, ), initializer = tf.zeros_initializer(), trainable=False)
        self.logvar_unif_prior = self.add_weight(shape=(self.latent_dim, ), initializer = tf.constant_initializer(10), trainable=False)
        
    def call(self, x, z):
        post_mu, post_logvar = self.encoder_old(x)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_unif_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g+tf.math.log(w) for g, w in zip(gaussians, self.mixture_weights)]
        
        density = tf.stack(gaussians, -1)
        return tf.math.log(tf.reduce_sum(tf.exp(density), -1)) # logsumexp