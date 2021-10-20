


#%% CRPS and CNNs

#%%
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import models,layers
import tensorflow_probability as tfp
tfd = tfp.distributions

import math
import xarray as xr
from scipy.stats import pearsonr,gamma
import os
import gc
import random


# %% CRPS of CSG distribution , following Grönquist et al. (2021)

class loss_CRPS_CSGD_V2(keras.losses.Loss):

    def __init__(self ): #, name="loss_CRPS_CSGD" 
        super().__init__() 
 
    def call(self, obs, par_mat):

        shift = par_mat[:,0]  - 1e-6 #negative value
        mu = par_mat[:,1]
        sigma = par_mat[:,2]

        shape = (mu/sigma)**2
        scale = (sigma**2)/mu
        rate = 1/scale
     
        betaf = tf.exp ( tfp.math.lbeta(0.5, (shape)+0.5) )
        dist_k = tfd.Gamma(shape, rate )  
        dist_kp1 = tfd.Gamma(shape+1 , rate )  
        dist_2k = tfd.Gamma(shape*2 , rate )  
        Fyk = dist_k.cdf( obs-shift )
        Fck = dist_k.cdf( -shift )
        FykP1 = dist_kp1.cdf(obs-shift )
        FckP1 = dist_kp1.cdf( -shift )
        F2c2k = dist_2k.cdf( 2*(-shift) )

        crps_scaled = (obs-shift)*(2.*Fyk-1.) + shift*(Fck**2) + shape*scale*(1.+2.*Fck*FckP1-(Fck**2)-2*FykP1) \
            - (shape*scale/(math.pi))*betaf*(1.-F2c2k)
        
        ind_notnan = tf.where( ~tf.math.is_nan(crps_scaled) ) #I add this to avoid use NaN CRPS values
        CRPS_notnan = ( tf.reduce_mean( tf.gather_nd(crps_scaled,ind_notnan) ) )

        return CRPS_notnan
        
#get an object
loss_CRPS_CSGD_object_V2 = loss_CRPS_CSGD_V2()




#%%
def ANN3out_2emb(image_size, n_channel, n_word_max_lat,n_word_max_lon, n_out_emb_dim, apply_dropout=False):
    #conv kernel size, conv channel, input words dimension, embedding output dimension
    ##lat, lon embedding separately


    n_feature = [ 64,  32, 16 ]
    initializer = tf.random_normal_initializer( 0, 0.01)

    lat_in = layers.Input(shape=(1,))#lat embed
    x1 = layers.Embedding(input_dim=n_word_max_lat, output_dim = n_out_emb_dim, input_length=1 )(lat_in) 
    lat_emb = layers.Flatten()(x1)

    lon_in = layers.Input(shape=(1,))#lon embed
    x2 = layers.Embedding(input_dim=n_word_max_lon, output_dim = n_out_emb_dim, input_length=1 )(lon_in) 
    lon_emb = layers.Flatten()(x2)

    #----------------

    #image input:
    features_in = layers.Input( shape=( image_size,image_size,n_channel) ) 
    features_in_vec = layers.Flatten()(features_in)    #flatten convolution results


    #combine 
    input_combine = layers.Concatenate()([features_in_vec, lat_emb, lon_emb] )
    x = layers.Dense(n_feature[0], kernel_initializer=initializer)( input_combine)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(n_feature[1], kernel_initializer=initializer)( x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(n_feature[2], kernel_initializer=initializer)( x)
    x = layers.BatchNormalization()(x)
    
    if (apply_dropout):
        x =layers.Dropout(0.5)(x)


    #OUTPUT LAYER
    x = layers.Dense(3, kernel_initializer=initializer)(x)




    #multi inputs
    model = keras.Model([lat_in, lon_in, features_in], x )



    return model
#%%
def lenet3out_1conv_2emb(image_size, n_channel, n_word_max_lat,n_word_max_lon, n_out_emb_dim, apply_dropout=False):
    #conv kernel size, conv channel, input words dimension, embedding output dimension
    ##lat, lon embedding separately

    initializer = tf.random_normal_initializer( 0, 0.01)

    lat_in = layers.Input(shape=(1,))#lat embed
    x1 = layers.Embedding(input_dim=n_word_max_lat, output_dim = n_out_emb_dim, input_length=1 )(lat_in) 
    lat_emb = layers.Flatten()(x1)

    lon_in = layers.Input(shape=(1,))#lon embed
    x2 = layers.Embedding(input_dim=n_word_max_lon, output_dim = n_out_emb_dim, input_length=1 )(lon_in) 
    lon_emb = layers.Flatten()(x2)


    #----------------
    #image input:
    f = [  64  ]
    kernel_size = 3

    features_in = layers.Input( shape=( image_size,image_size,n_channel) ) 
    # Conv => ReLu => Pool
    x = layers.Conv2D( filters=f[0], kernel_size=kernel_size, strides=1, activation='elu',padding='valid', kernel_initializer=initializer)(features_in)
    # res.add(layers.MaxPooling2D( pool_size=(2,2), strides=2, padding='valid'))
    x = layers.BatchNormalization()(x)

    if (apply_dropout):
        x =layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)    #flatten convolution results

    #combine 
    input_combine = layers.Concatenate()([x, lat_emb, lon_emb] )

    #Dense 1
    x = layers.Dense(6, kernel_initializer=initializer)( input_combine)
    #Dense 2
    x = layers.Dense(3, kernel_initializer=initializer)( x)



    #multi inputs
    model = keras.Model([lat_in, lon_in, features_in], x )

    return model




#%%
def lenet3out_2conv_2emb(image_size, n_channel, n_word_max_lat,n_word_max_lon, n_out_emb_dim, apply_dropout=False):
    #conv kernel size, conv channel, input words dimension, embedding output dimension
    ##lat, lon embedding separately

    initializer = tf.random_normal_initializer( 0, 0.01)
    
    lat_in = layers.Input(shape=(1,))#lat embed
    x1 = layers.Embedding(input_dim=n_word_max_lat, output_dim = n_out_emb_dim, input_length=1 )(lat_in) 
    lat_emb = layers.Flatten()(x1)

    lon_in = layers.Input(shape=(1,))#lon embed
    x2 = layers.Embedding(input_dim=n_word_max_lon, output_dim = n_out_emb_dim, input_length=1 )(lon_in) 
    lon_emb = layers.Flatten()(x2)


    #----------------
    #image input:
    # f = [  8, 4 ]
    # f = [ 32, 16  ]
    f = [ 64,  32  ]
    kernel_size = 3

    features_in = layers.Input( shape=( image_size,image_size,n_channel) ) 
    # Conv => ReLu 
    x = layers.Conv2D( filters=f[0], kernel_size=kernel_size, strides=1, activation='elu',padding='valid', kernel_initializer=initializer)(features_in)
    x = layers.BatchNormalization()(x)
    # Conv => ReLu 
    x = layers.Conv2D( filters=f[1], kernel_size=kernel_size, strides=1, activation='elu',padding='valid', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)


    if (apply_dropout):
        x =layers.Dropout(0.5)(x)


    x = layers.Flatten()(x)    #flatten convolution results

    #combine 
    input_combine = layers.Concatenate()([x, lat_emb, lon_emb] )
    x = layers.Dense(6, kernel_initializer=initializer)( input_combine)
    #Dense 2
    x = layers.Dense(3, kernel_initializer=initializer)( x)

    #multi inputs
    model = keras.Model([lat_in, lon_in, features_in], x )

    return model



#%%
def lenet3out_3conv_2emb(image_size, n_channel, n_word_max_lat,n_word_max_lon, n_out_emb_dim, apply_dropout=False):
    #conv kernel size, conv channel, input words dimension, embedding output dimension
    ##lat, lon embedding separately

    initializer = tf.random_normal_initializer( 0, 0.01)
    
    lat_in = layers.Input(shape=(1,))#lat embed
    x1 = layers.Embedding(input_dim=n_word_max_lat, output_dim = n_out_emb_dim, input_length=1 )(lat_in) 
    lat_emb = layers.Flatten()(x1)

    lon_in = layers.Input(shape=(1,))#lon embed
    x2 = layers.Embedding(input_dim=n_word_max_lon, output_dim = n_out_emb_dim, input_length=1 )(lon_in) 
    lon_emb = layers.Flatten()(x2)


    #----------------
    #image input:
    f = [ 64,  32, 16  ]
    kernel_size = 3

    features_in = layers.Input( shape=( image_size,image_size,n_channel) ) 
    # Conv => ReLu 
    x = layers.Conv2D( filters=f[0], kernel_size=kernel_size, strides=1, activation='elu',padding='valid', kernel_initializer=initializer)(features_in)
    x = layers.BatchNormalization()(x)
    # Conv => ReLu 
    x = layers.Conv2D( filters=f[1], kernel_size=kernel_size, strides=1, activation='elu',padding='valid', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    # Conv => ReLu 
    x = layers.Conv2D( filters=f[2], kernel_size=kernel_size, strides=1, activation='elu',padding='valid', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)


    if (apply_dropout):
        x =layers.Dropout(0.5)(x)


    x = layers.Flatten()(x)    #flatten convolution results

    #combine 
    input_combine = layers.Concatenate()([x, lat_emb, lon_emb] )
    x = layers.Dense(6, kernel_initializer=initializer)( input_combine)
    #Dense 2
    x = layers.Dense(3, kernel_initializer=initializer)( x)

    #multi inputs
    model = keras.Model([lat_in, lon_in, features_in], x )

    return model


#%%
def lenet3out_4conv_2emb(image_size, n_channel, n_word_max_lat,n_word_max_lon, n_out_emb_dim, apply_dropout=False):
    #conv kernel size, conv channel, input words dimension, embedding output dimension
    ##lat, lon embedding separately

    initializer = tf.random_normal_initializer( 0, 0.01)
    
    lat_in = layers.Input(shape=(1,))#lat embed
    x1 = layers.Embedding(input_dim=n_word_max_lat, output_dim = n_out_emb_dim, input_length=1 )(lat_in) 
    lat_emb = layers.Flatten()(x1)

    lon_in = layers.Input(shape=(1,))#lon embed
    x2 = layers.Embedding(input_dim=n_word_max_lon, output_dim = n_out_emb_dim, input_length=1 )(lon_in) 
    lon_emb = layers.Flatten()(x2)


    #----------------
    #image input:
    # f = [  8, 4 ]
    # f = [ 32, 16  ]
    f = [ 64,  32, 16, 8  ]
    kernel_size = 3

    features_in = layers.Input( shape=( image_size,image_size,n_channel) ) 
    # Conv => ReLu 
    x = layers.Conv2D( filters=f[0], kernel_size=kernel_size, strides=1, activation='elu',padding='valid', kernel_initializer=initializer)(features_in)
    x = layers.BatchNormalization()(x)
    # Conv => ReLu 
    x = layers.Conv2D( filters=f[1], kernel_size=kernel_size, strides=1, activation='elu',padding='valid', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    # Conv => ReLu 
    x = layers.Conv2D( filters=f[2], kernel_size=kernel_size, strides=1, activation='elu',padding='valid', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    # Conv => ReLu 
    x = layers.Conv2D( filters=f[3], kernel_size=kernel_size, strides=1, activation='elu',padding='valid', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)


    if (apply_dropout):
        x =layers.Dropout(0.5)(x)


    x = layers.Flatten()(x)    #flatten convolution results

    #combine 
    input_combine = layers.Concatenate()([x, lat_emb, lon_emb] )
    x = layers.Dense(6, kernel_initializer=initializer)( input_combine)
    #Dense 2
    x = layers.Dense(3, kernel_initializer=initializer)( x)

    #multi inputs
    model = keras.Model([lat_in, lon_in, features_in], x )

    return model

