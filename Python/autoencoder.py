'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
#from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import os.path as path
import argparse
import glob 
from PIL import Image
import matplotlib.image as mpimg

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 64 # TODO change here if 
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

########################################################################################################################
# Export the frozen graph for later use in Unity

def export_model(saver, model, model_name, input_node_names, output_node_name):
    if not path.exists('out'):
        os.mkdir('out')

    tf.train.write_graph(K.get_session().graph_def, 'out', model_name + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + model_name + '.chkp')

    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None, False,
                              'out/' + model_name + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              'out/frozen_' + model_name + '.bytes', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + model_name + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

########################################################################################################################

def create_hieroglyphs():
    # Create Hieroglyph data set
    imgs = glob.glob('fonts/Hieroglyphs/*.png')
    x_train = []
    y_train = []
    for i in range(len(imgs)):

        # load image
        img = Image.open(imgs[i])
        img.thumbnail((64, 64), Image.NEAREST) # resizes image in-place
        img.convert('L')
        
        # rotate image
        for r in np.linspace(-15,15,21):
            rimg = img.rotate(r, fillcolor='black')
            data = list(rimg.getdata())

            greyscale = np.zeros(len(data))
            for j in range(len(data)):
                greyscale[j] = data[j][0]/255

            # shift along each axis
            for sx in np.arange(-8,8):

                for sy in np.arange(-4,4):
                    gimg = np.copy(greyscale).reshape((64,64))
                    rolled = np.roll( np.roll(gimg, sx), sy)

                    # new training point
                    x_train.append( rolled.flatten() )
                    y_train.append( i )

    # inputs to vae
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    #x_train = preprocessing.scale(x_train, axis=1)

    return (x_train, y_train), (x_train, y_train) # TODO generate test data


def build_encoder(input_size,latent_dim):
     
    # build encoder model
    inputs = Input(shape=(input_size,), name='encoder_input')
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)


    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def sampling(args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """
        args = z_mean, z_log_var
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    return encoder, inputs, z_mean, z_log_var

def build_decoder(output_size,latent_dim):

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(256, activation='relu', name='decoder_middle1')(latent_inputs)
    x = Dense(1024, activation='relu', name='decoder_middle2')(x)
    outputs = Dense(output_size, activation='sigmoid', name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    return decoder, outputs

def build_vae(models, inputs, outputs, z_mean, z_log_var, mse_bool, name):
    
    encoder, decoder = models 
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name=name)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if mse_bool:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= encoder.input_shape[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var) # error to keep within distribution
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    return vae


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()


    # load data 
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) =  create_hieroglyphs()
    data = (x_train, y_train)

    # create encoder and decoder
    latent_dim = 2
    encoder, inputs, z_mean, z_log_var = build_encoder(x_train.shape[1], latent_dim)
    decoder, outputs = build_decoder(x_train.shape[1], latent_dim)
    
    # create vae 
    models = ( encoder, decoder )
    vae = build_vae(models, inputs, outputs, z_mean, z_log_var, args.mse, 'vae_mlp_hiero')

    # load weights from file 
    if args.weights:
        vae.load_weights(args.weights)
    else:

        # train the autoencoder
        batch_size = 32
        epochs = 10

        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_train, None))

        vae.save_weights('vae_mlp_hiero.h5')

    # nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]

    # export for unity compatibility
    models[1].save_weights("decoder_weights")
    decoder, outputs = build_decoder(x_train.shape[1], latent_dim)
    decoder.load_weights("decoder_weights")
    export_model(tf.train.Saver(), decoder,"hiero_decoder_4096", ['decoder_input'], "decoder_output/Sigmoid")

    
    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_hiero")

    # interesting tutorials 
    # https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/Using-TensorFlow-Sharp-in-Unity-(Experimental).md
    # https://github.com/migueldeicaza/TensorFlowSharp




