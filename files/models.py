import tensorflow as tf
import numpy as np

# Model
class DCGANConv(tf.keras.layers.Layer):
    # Conv block to be used in DCGAN-style encoder
    # Each block consists of: Conv2D - BatchNorm - LeakyReLU
    def __init__(self, out_filter):
        super().__init__()
        self.conv2d = tf.keras.layers.Conv2D(out_filter, 
                                             (4, 4), 
                                             strides=(2, 2), 
                                             padding='same')
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=3)
        self.leakyrelu = tf.keras.layers.LeakyReLU(0.2)
        
    def call(self, x):
        x = self.conv2d(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x
    
class DCGANConvT(tf.keras.layers.Layer):
    # ConvTranspose block to be used in DCGAN-style decoder
    # Each block consists of: Conv2DTranspose - BatchNorm - LeakyReLU
    def __init__(self, out_filter):
        super().__init__()
        self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(out_filter, 
                                                               (4, 4), 
                                                               strides=(2, 2), 
                                                               padding='same')
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=3)
        self.leakyrelu = tf.keras.layers.LeakyReLU(0.2)
        
    def call(self, x):
        x = self.conv2dtranspose(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x

class LSTMCellGaussian(tf.keras.layers.Layer):
    # This LSTM cell outputs Gaussian parameters
    # This cell will be used in inference: variational distribution q, and in prior. 
    # Letting LSTM input dim equal hidden dim is by design of the original paper.
    # It consists of Dense - LSTM - Dense.
    def __init__(self, encoder_dim, lstm_dim, latent_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(lstm_dim, input_shape=(encoder_dim,))
        self.lstm_cell = tf.keras.layers.LSTMCell(lstm_dim)
        self.dense2 = tf.keras.layers.Dense(2*latent_dim)
        
    def call(self, x, states):
        # Run through the layers:
        x = self.dense1(x)
        x, states = self.lstm_cell(x, states)
        x = self.dense2(x)
        # Split final dense layers output into two Gaussian parameters: mu, logvar
        mu, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mu, logvar, states[0], states[1]
    
class LSTMCellDec(tf.keras.layers.Layer):
    # This LSTM cell outputs a representation to be decoded into an image.
    # Letting LSTM input dim equal hidden dim is by design of the original paper.
    # It consists of Dense - LSTM - LSTM - Dense.
    def __init__(self, rep_dim, lstm_dim, output_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(lstm_dim, input_shape=(rep_dim,))
        self.lstm_cell1 = tf.keras.layers.LSTMCell(lstm_dim)
        self.lstm_cell2 = tf.keras.layers.LSTMCell(lstm_dim)
        self.lstm_cell = tf.keras.layers.StackedRNNCells([self.lstm_cell1, self.lstm_cell2])
        self.dense2 = tf.keras.layers.Dense(output_dim, activation='tanh')
        
    def call(self, x, states):
        x = self.dense1(x)
        x, states = self.lstm_cell(x, states)
        x = self.dense2(x)
        return x, states[0], states[1]

class Encoder(tf.keras.layers.Layer):
    # DCGAN-style encoder
    # It uses the previously defined DCGANConv blocks.
    def __init__(self, output_dim, filter_size_init=64):
        super().__init__()
        self.nf = filter_size_init
        self.dcgan_conv1 = DCGANConv(self.nf)
        self.dcgan_conv2 = DCGANConv(self.nf*2)
        self.dcgan_conv3 = DCGANConv(self.nf*4)
        self.dcgan_conv4 = DCGANConv(self.nf*8)
        self.conv_final = tf.keras.layers.Conv2D(output_dim, 
                                                 (4, 4), 
                                                 strides=(1, 1), 
                                                 padding='valid',
                                                 activation='tanh')
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=3)
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, x):
        x1 = self.dcgan_conv1(x)
        x2 = self.dcgan_conv2(x1)
        x3 = self.dcgan_conv3(x2)
        x4 = self.dcgan_conv4(x3)
        x5 = self.conv_final(x4)
        x5 = self.batchnorm(x5)
        x5 = self.flatten(x5)
        # We return [x1,x2,x3,x4] as well because they might be used in skip connections
        return x5, [x1,x2,x3,x4]

class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, input_dim, filter_size=64, output_channel=1, use_skip=True):
        super().__init__()
        self.nf = filter_size # filter size of the second-to-last conv transpose layer
        self.nc = output_channel # 1 for SM-MNIST
        self.use_skip = use_skip # whether skip-connections are used
        
        self.reshape = tf.keras.layers.Reshape((1, 1, input_dim))
        self.conv_t1 = tf.keras.layers.Conv2DTranspose(self.nf*8, 
                                                       (4, 4), 
                                                       strides=(1, 1), 
                                                       padding='valid')
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=3)
        self.leakyrelu = tf.keras.layers.LeakyReLU(0.2)
        self.dcgan_conv_t1 = DCGANConvT(self.nf*4)
        self.dcgan_conv_t2 = DCGANConvT(self.nf*2)
        self.dcgan_conv_t3 = DCGANConvT(self.nf)
        self.conv_t2 = tf.keras.layers.Conv2DTranspose(self.nc, 
                                                       (4, 4), 
                                                       strides=(2, 2), 
                                                       padding='same')
        
    def call(self, x, skip):
        x = self.reshape(x)
        x = self.conv_t1(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        # Concatenate with skip connections if we use it
        if self.use_skip:
            x = self.dcgan_conv_t1(tf.concat([x, skip[3]], axis=-1))
            x = self.dcgan_conv_t2(tf.concat([x, skip[2]], axis=-1))
            x = self.dcgan_conv_t3(tf.concat([x, skip[1]], axis=-1))
            x = self.conv_t2(tf.concat([x, skip[0]], axis=-1))
        # If we do not use skip connections
        else:
            x = self.dcgan_conv_t1(x)
            x = self.dcgan_conv_t2(x)
            x = self.dcgan_conv_t3(x)
            x = self.conv_t2(x)
        return x
    
# It shall be used as RNN Cell wrapped inside TensorFlow RNN class
# It incorporates all modules
class SVGCell(tf.keras.layers.Layer):
    
    def __init__(self, encoder_dim, lstm_q_dim, lstm_prior_dim, latent_dim, 
                 lstm_dec_dim, lstm_dec_out_dim, use_skip=True):
        super().__init__()
        # Necessary attribute of TensorFlow Cell class; we have 4 lstms:
        self.state_size = [[lstm_q_dim, lstm_q_dim], [lstm_prior_dim, lstm_prior_dim], 
                           [lstm_dec_dim, lstm_dec_dim], [lstm_dec_dim, lstm_dec_dim]]
        # Here we initialize all the modules of the architecture:
        self.lstm_q = LSTMCellGaussian(encoder_dim, lstm_q_dim, latent_dim)
        self.lstm_prior = LSTMCellGaussian(encoder_dim, lstm_prior_dim, latent_dim)
        self.encoder = Encoder(encoder_dim)
        self.lstm_dec = LSTMCellDec(latent_dim+encoder_dim, lstm_dec_dim, lstm_dec_out_dim)
        self.decoder = Decoder(lstm_dec_out_dim, use_skip=use_skip)
        # Implementational things:
        self.batch_starts = False
        self.x_tm1 = tf.random.normal((256, 64, 64, 1), 0.0, 0.1)
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def call(self, x, states):
        # Expand out LSTM states
        lstm_q_states, lstm_prior_states, lstm_dec_1, lstm_dec_2 = states
        lstm_q_h, lstm_q_c = lstm_q_states
        lstm_prior_h, lstm_prior_c = lstm_prior_states
        
        # ------- Inference and generation begins -------+
        # Encoder acting on preivous frame
        enc_out_tm1, skip = self.encoder(self.x_tm1)
        # Encoder acting on current frame
        enc_out_t = self.encoder(x)[0]
        # LSTM_q, the LSTM in the variational distribution (inference)
        mean, logvar, lstm_q_h, lstm_q_c = self.lstm_q(enc_out_t, [lstm_q_h, lstm_q_c])
        # LSTM_prior, the LSTM as part of the prior distribution
        mean_0, logvar_0, lstm_prior_h, lstm_prior_c = self.lstm_prior(enc_out_tm1, [lstm_prior_h, lstm_prior_c])
        # Latent variables, using output from encoder + LSTM
        z = self.reparameterize(mean, logvar)
        # Decode: the generative model uses two LSTMs and a DCGAN decoder
        concat = tf.concat([enc_out_tm1, z], axis=1)
        lstm_out, lstm_dec_1, lstm_dec_2 = self.lstm_dec(concat, [lstm_dec_1, lstm_dec_2])
        x_recons = self.decoder(lstm_out, skip)
        # ------- Inference and generation ends -------+
        
        # Set self.x_tm1 to x, so that when this function is called again, 
        # self.x_tm1 will be the current x (or previous x from the perspective of next call).
        # The if-condition is here because the system calls this the cell exactly once before actual training.
        # We don't want to update self.x_tm1 in this particular call (self.batch_starts is initialized to False).
        if self.batch_starts:
            self.x_tm1 = x
        self.batch_starts = True
        
        # Return a list of useful outputs, and a list of LSTM states (necessary TensorFlow format)
        return [mean, logvar, mean_0, logvar_0, z, x_recons], \
               [[lstm_q_h, lstm_q_c], [lstm_prior_h, lstm_prior_c], lstm_dec_1, lstm_dec_2]
    
    # Generate; similar to call() but has differences
    def generate(self, x, states):
        # Expand out LSTM states
        lstm_q_states, lstm_prior_states, lstm_dec_1, lstm_dec_2 = states
        lstm_q_h, lstm_q_c = lstm_q_states
        lstm_prior_h, lstm_prior_c = lstm_prior_states
        # Encoder acting on preivous frame
        enc_out_tm1, skip = self.encoder(x)
        # LSTM_prior, the LSTM as part of the prior distribution
        mean_0, logvar_0, lstm_prior_h, lstm_prior_c = self.lstm_prior(enc_out_tm1, [lstm_prior_h, lstm_prior_c])
        # Latent variables, using output from encoder + LSTM
        z = self.reparameterize(mean_0, logvar_0)
        # Decode: the generative model uses two LSTMs and a DCGAN decoder
        concat = tf.concat([enc_out_tm1, z], axis=1)
        lstm_out, lstm_dec_1, lstm_dec_2 = self.lstm_dec(concat, [lstm_dec_1, lstm_dec_2])
        x_out = self.decoder(lstm_out, skip)
        
        return x_out, [[lstm_q_h, lstm_q_c], [lstm_prior_h, lstm_prior_c], lstm_dec_1, lstm_dec_2]
        
