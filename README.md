# Unity-Variational-Autoencoder
A variational autoencoder made with tensorflow loaded into Unity for the procedural generation of 3D surfaces

![](https://github.com/pearsonkyle/Unity-Vartiational-Autoencoder/blob/master/MNIST_VAE.gif)


## Unity 
To change the tensorflow model in unity:
1. Train a new model in tensorflow (e.g. VAE in [autoencoder.py](https://github.com/pearsonkyle/Unity-Variational-Autoencoder/blob/master/Python/autoencoder.py#L242))

2. Load the frozen tensor flow graph into unity 
![](https://github.com/pearsonkyle/Unity-Variational-Autoencoder/blob/master/tensorflow_model_unity.png)

3. Change the resolution of the tensorflow model in Unity
![](https://github.com/pearsonkyle/Unity-Variational-Autoencoder/blob/master/meshgen_edit.png)


## Python
The variational autoencoder is created using keras and tensorflow in Python3.6. Please see the Python/ directory

Example code from: [autoencoder.py](https://github.com/pearsonkyle/Unity-Variational-Autoencoder/blob/master/Python/autoencoder.py#L242)
```python
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
    export_model(tf.train.Saver(), decoder,"hiero2_decoder16384", ['decoder_input'], "decoder_output/Sigmoid")
```
Make sure in the build_decoder function the layers have names so that we can reference them from the tensorflow graph. 


## Useful links
- https://github.com/llSourcell/Unity_ML_Agents/tree/master/docs
- https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/Using-TensorFlow-Sharp-in-Unity-(Experimental).md
