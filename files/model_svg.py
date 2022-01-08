import tensorflow as tf 
import numpy as np 
from files.utils import *
from files.models import *


# The highest level class that can:
# - have an attribute `self.svg_layer` that wraps around SVGCell
# - compute loss
# - train
# - generate
class SVG(tf.keras.Model):
    
    def __init__(self, encoder_dim, lstm_q_dim, lstm_prior_dim, latent_dim, 
                 lstm_dec_dim, lstm_dec_out_dim,
                 use_skip=True, beta=1e-4, num_conditioned_frame=5, num_frame=20):
        super().__init__()
        self.beta = beta
        self.num_conditioned_frame = num_conditioned_frame
        self.num_frame = num_frame
        self.num_generate_frame = self.num_frame-self.num_conditioned_frame
        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=(None, 64, 64, 1))
        
        # SVGCell
        self.svg_cell = SVGCell(encoder_dim, lstm_q_dim, lstm_prior_dim, latent_dim, 
                                 lstm_dec_dim, lstm_dec_out_dim, use_skip=use_skip)
        # Wraps around the SVGCell
        self.svg_layer = tf.keras.layers.RNN(self.svg_cell, 
                                             return_sequences=True, 
                                             return_state=True)   
        
    def call(self, x):
        # Runs through the whole video.
        # We only need to call the wrapper self.svg_layer.
        x = tf.cast(x, tf.float32)
        x = self.input_layer(x)
        outputs = self.svg_layer(x)
        return outputs
    
    def compute_loss(self, x, svg_layer_outputs):
        
        # Expand out:
        mean, logvar, mean_0, logvar_0, z, x_recons, \
        state1, state2, state3, state4 = svg_layer_outputs
        
        # Commented out is for Bernoulli r.v.
        # Original paper uses squared loss, somewhat like Gaussian r.v., instead.
#         cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
#             logits=x_recons, 
#             labels=x)
#         logpx_z = -tf.reduce_sum(cross_ent, axis=list(range(2, len(x.shape))))

        # Compute log p(x|z):
        squared_loss = (x_recons - x) ** 2
        logpx_z = -tf.reduce_mean(squared_loss, axis=list(range(2, len(x.shape))))
        # Compute log p(z):
        logpz = log_normal_pdf(z, mean_0, logvar_0, raxis=list(range(2, len(z.shape))))
        # Compute log q(z|x):
        logqz_x = log_normal_pdf(z, mean, logvar, raxis=list(range(2, len(z.shape))))
        # ELBO, average across batches to get a number-of-frames-long vector:
        total_loss_frames = -tf.reduce_mean(logpx_z + self.beta*(logpz - logqz_x), axis=0)
        # Gather the frames that we are not conditioning on, and sum into final loss:
        loss = tf.reduce_sum(
            tf.gather(total_loss_frames, list(range(self.num_conditioned_frame-1, self.num_frame-1))))
        
        return loss
    
    # Train
    def train(self, train_dataset, test_dataset, epochs=10, lr=1e-3, batch_size=256, test_generate_batch_size=10):
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        self.ssim = np.zeros((epochs, test_generate_batch_size, self.num_generate_frame))
        self.loss_list = []
        self.batch_size = batch_size
        self.test_generate_batch_size = test_generate_batch_size

        # Training starts
        for epoch in range(epochs):
            
            print("\nStart of epoch %d" % (epoch,))
            self.epoch = epoch
            loss_across_batches = 0

            for step, x_batch in enumerate(train_dataset):
                
                # Initialize with the first frame and train on the rest
                self.svg_cell.x_tm1 = x_batch[:,0,:,:,:]
                x_batch = x_batch[:,1:,:,:,:]

                # Train
                with tf.GradientTape() as tape:

                    out = self(x_batch)
                    loss_value = self.compute_loss(x_batch, out)
                
                # Computes gradients and updates variables
                grads = tape.gradient(loss_value, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                
                # Record and see results, plus some implementational stuff
                loss_across_batches += loss_value.numpy()
                self.svg_cell.batch_starts = False

                if step % 5 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))

                if step % 30 == 0 and step != 0:
                    x_recons = out[5]
                    print('Original training video (19 frames)')
                    fig = plt.figure(figsize=(16, 4))
                    plt.subplots_adjust(wspace=0.1, hspace=0)
                    for i in range(19):
                        plt.subplot(2, 10, i+1)
                        plt.imshow(x_batch[0,i,:,:,:], cmap='gray')
                        plt.axis('off')
                    plt.show()
                    print('Reconstructed training video (19 frames)')
                    fig = plt.figure(figsize=(16, 4))
                    plt.subplots_adjust(wspace=0.1, hspace=0)
                    for i in range(19):
                        plt.subplot(2, 10, i+1)
                        plt.imshow(x_recons[0,i,:,:,:], cmap='gray')
                        plt.axis('off')
                    plt.show()
                
            # Generate on the data:
            for x_batch in train_dataset.take(1):
                x_batch = x_batch[:test_generate_batch_size,:,:,:,:]
                self.generate(x_batch)  
            print('SSIM', np.mean(self.ssim[epoch,:,:]))
            
            # Save results:
            self.loss_list.append(loss_across_batches)
            if epoch % 10 == 0 and epoch != 0:
                model.save_weights('trained_models/model/model')
                np.savetxt('results/ssim.csv', np.reshape(self.ssim,(epochs,-1)))
                np.savetxt('results/losses.csv', self.loss_list)
            
            # --- An epoch ends ---+
           
    # ------- Train function ends -------+
                
    # Function to generate at test time; use generated image as input at next time-step
    def generate(self, x_batch):
        
        # Initialize with the first frame and condition on the rest like in training
        self.svg_cell.x_tm1 = x_batch[:,0,:,:,:]
        x_batch = x_batch[:,1:,:,:,:]
        
        # Split x_batch into conditioned ones, and ones that model doesn't know
        x_batch_cond, x_batch_gen = tf.split(
            x_batch, 
            num_or_size_splits=[self.num_conditioned_frame-1, 
                                self.num_generate_frame], 
            axis=1)
        
        # Run on the conditioned frames:
        out = self(x_batch_cond)
        mean, logvar, mean_0, logvar_0, z, x_recons, \
        lstm_q_states, lstm_prior_states, \
        lstm_dec_1_states, lstm_dec_2_states = out
        states = [lstm_q_states, lstm_prior_states, lstm_dec_1_states, lstm_dec_2_states]
        x_out = x_recons[:,self.num_conditioned_frame-2,:,:,:]
        
        # Plotting:
        print('\n------- Generating -------+\n')
        print('Original video (15 frames)')
        fig = plt.figure(figsize=(16, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        for t in range(self.num_generate_frame):
            plt.subplot(2, math.ceil(self.num_generate_frame/2), t+1)
            plt.imshow(x_batch_gen[0,t,:,:,:], cmap='gray')
            plt.axis('off')
        plt.show()
        fig = plt.figure(figsize=(16, 4))
        plt.subplots_adjust(wspace=0.1, hspace=0)
        print('Generated video (15 frames)')
        
        # Generating: loop across frames to generate new frames
        for t in range(self.num_generate_frame):
            x_out, states = self.svg_cell.generate(x_out, states)
            
            plt.subplot(2, math.ceil(self.num_generate_frame/2), t+1)
            plt.imshow(x_out[0,:,:,:], cmap='gray')
            plt.axis('off')
            for i in range(self.test_generate_batch_size):
                self.ssim[self.epoch, i, t] = ssim_metric(x_batch_gen.numpy()[i,t,:,:,0],
                                                          x_out.numpy()[i,:,:,0])
        plt.show()
            
        self.svg_cell.batch_starts = False
            
        return