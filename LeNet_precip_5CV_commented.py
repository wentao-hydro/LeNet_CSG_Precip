#%% 
# Here is the main script of a LeNet type CNN-based post-processing meethod for short-term precipitation forecasts 
# The CNN structures and CRPS are defined in CNN_aux.py

#from line 50 to line 130:  load obs, fcst and other dataset
#from line 233 to line 369: prepare TRAIN, VALID, TEST dataset 
#after line 369: train and validation steps

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
import copy
# import matplotlib.pyplot as plt

#%% import CNNs and CRPS from CNN_aux.py
from CNN_aux import *


# %% set random seed 
seed = 21
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# %%#setting 

ndaysummer=35

# shuffled_index = np.random.permutation(np.arange(19))
shuffled_index = [ 10, 17,  3, 16,  6,  9,  2, 15, 13, 18,  4,  7, 14,  5, 11,  8, 12, 0,  1]
#that series is a permutation result

index_all_list =  ( np.arange(0, 19) ).tolist()

file_path = '/data/home/scv0203/run/EC_0704/'



#%% load research region mask 
region_class = np.loadtxt( file_path+'input/dem_huaip25_0619.txt', skiprows = 6)
# plt.imshow( np.flipud(region_class) )

#% index for grids within the research region
region_east_index = np.where(np.flipud(region_class) >0 )
lat_index_used_FINAL = region_east_index[0]
lon_index_used_FINAL = region_east_index[1]
ngrid_used = len( lat_index_used_FINAL)



# %% load obs of 0.25 deg
obs_19982017_leadtime_0p25 = xr.open_dataarray (file_path+"input/obs_0p25_CMA_leadtime_19982017_China.nc")
obs_lat = obs_19982017_leadtime_0p25.latitude
obs_lon = obs_19982017_leadtime_0p25.longitude



#%% select obs of the research region
lat_min,lat_max = 31,31+0.25*(21-1) #region
lon_min,lon_max = 112,112+0.25*(38-1-1)
ind_lat_obs = np.where( (obs_lat >= lat_min)&(obs_lat <= lat_max) )[0]
ind_lon_obs = np.where( (obs_lon >= lon_min)&(obs_lon <= lon_max) )[0]
nlon_obs = len(ind_lon_obs)
nlat_obs = len(ind_lat_obs)
obs_lat_used = obs_lat[ind_lat_obs[:nlat_obs]]
obs_lon_used = obs_lon[ind_lon_obs[:nlon_obs]]
obs_lat_min = np.min(obs_lat)
obs_lon_min = np.min(obs_lon)



#%% output
predictmean_array = np.zeros(( ndaysummer , 19,  ngrid_used))
predictens_array= np.zeros(( ndaysummer , 19,  ngrid_used, 100))

rawfcst_array = np.zeros(( ndaysummer , 19,  ngrid_used))
y_true_array = np.zeros(( ndaysummer , 19,  ngrid_used))



#%% 

for ilead in range( 2 ):


    #%%  load fcst, pressure level var, each center grid with 9*9 neighbor grids
    fcst_PL_array = xr.open_dataarray( file_path +'input/PL_fcst_9by9_Huaihe_lead'+str(ilead)+'_0704.nc' )
    #  dimension: 35day , 19 year, ngrid_used, nlon_win=9,nlon_win=9 ,n_variable (predictor)
    #predictors include:   Z U V Q W D T at 200,500,850hPa


    #%%  load fcst, surface var, each center grid with 9*9 neighbor grids
    fcst_surf_array = xr.open_dataarray( file_path +'input/surface_fcst_9by9_Huaihe_lead'+str(ilead)+'.nc' )
    #  input dimension: 35day , 19 year, ngrid_used, nlon_win=9,nlon_win=9 ,n_variable (predictor)
    #  predictors include: TP,CP,  TCW, TCWV,  TCC, MSL,  10U/V,  2m T


    #%% select obs lead 1 and 2 days 
    obs_region = obs_19982017_leadtime_0p25[35:70, :, 1:, ind_lat_obs , ind_lon_obs ] # dimension: days,years,lead,lat,lon
    obs_region = np.array(obs_region)
    obs_used = obs_region [:, :, ilead, lat_index_used_FINAL , lon_index_used_FINAL ]  #select grids in the region,   


    #%% load DEM, each center grid with 9*9 neighbor grids
    DEM_neighbor = xr.open_dataarray(file_path+"input/DEM_neighbor_0p25_region_9neighbor_0617.nc")
    # dimension: nlat, nlon, windows dimension of lat, windows dimension of lon

    DEM_lat = DEM_neighbor.latitude
    DEM_lon = DEM_neighbor.longitude
    #% select a smaller rectangle region
    ind_lat_fcst = np.where( (DEM_lat >= lat_min)&(DEM_lat <= lat_max) )[0]
    ind_lon_fcst = np.where( (DEM_lon >= lon_min)&(DEM_lon <= lon_max) )[0]
    DEM_part =  (DEM_neighbor[ ind_lat_fcst , ind_lon_fcst, (4-win_half):(4+win_half+1), (4-win_half):(4+win_half+1) ])
    DEM_part = np.array(DEM_part)
    #% select grids in research region
    DEM_neighbor0p25_mat = DEM_part [lat_index_used_FINAL , lon_index_used_FINAL,: ,: ]  
    DEM_neighbor.close()



    #-------------------
    #!! loop through different sizes of input layer, 
    # e.g., win_half=2 means 5*5 input layer size; win_half=3 means 7*7 input layer size;  
    # (should be distinguish from conv. kernel size, which is 3*3 through all tests here)
    for  win_half in range( 3,4 ): 

        nlat_win = 2*win_half+1 
        nlon_win = 2*win_half+1
        n_input_size = nlat_win
        flag_DEM=True
        print(win_half)
        
        if (win_half==1):
            EPOCHS= 40
        else:
            EPOCHS= 60
        learning_rate = 0.001/10 



 

        #--------------------------------------------------- 
        #select index of pressure level predictors used in the model
        varlist_used = [2,5,8,11,14,17,20,   1,4,7,10,13,16, 19,     0,3,6,9,12,15,18 ] #850 + 500 + 200
        n_var_PL = len(varlist_used)
        n_var_SURF= 9 #use all 9 surface predictors
        n_channel=n_var_SURF + n_var_PL + 1  #  add 1 for elevation predictor (DEM)
                        

        #select input layer size  (not conv. kernel size! )
        if n_input_size ==1: #1*1 input, i.e.,fully conneted network
            methodname =  'ANNCSGD_'+str(n_input_size)+'in-64-32-16-3-STDpre-Surf850 500 200-DEM-Emb_HH_lead'+str(ilead )+'_0712'
            model =  ANN3out_2emb(n_input_size, n_channel, max(lat_index_used_FINAL)+1, max(lon_index_used_FINAL)+1,  1) 
            print(model.summary() )

        elif n_input_size ==3: #3*3 input
            methodname =  'LenetCSGD_'+str(n_input_size)+'in_1conv-64-2hid_STDpre-Surf850 500 200-DEM-Emb_HH_lead'+str(ilead )+'_0712'
            model =  lenet3out_1conv_2emb(n_input_size, n_channel, max(lat_index_used_FINAL)+1, max(lon_index_used_FINAL)+1,  1) 
            print(model.summary() )

        elif n_input_size ==5: #5*5 input
            methodname =  'LenetCSGD_'+str(n_input_size)+'in_2conv-64-2hid_STDpre-Surf850 500 200-DEM-Emb_HH_lead'+str(ilead )+'_0712'
            model =  lenet3out_2conv_2emb(n_input_size, n_channel, max(lat_index_used_FINAL)+1, max(lon_index_used_FINAL)+1,  1) 
            print(model.summary() )            

        elif n_input_size ==7: #7*7 input
            methodname =  'LenetCSGD_'+str(n_input_size)+'in_3conv-64-2hid_STDpre-Surf850 500 200-DEM-Emb_HH_lead'+str(ilead )+'_0712'
            model =  lenet3out_3conv_2emb(n_input_size, n_channel, max(lat_index_used_FINAL)+1, max(lon_index_used_FINAL)+1,  1) 
            print(model.summary() )  

        elif n_input_size ==9: #9*9 input  
            methodname =  'LenetCSGD_'+str(n_input_size)+'in_4conv-64-2hid_STDpre-Surf850 500 200-DEM-Emb_HH_lead'+str(ilead )+'_0712'
            model =  lenet3out_4conv_2emb(n_input_size, n_channel, max(lat_index_used_FINAL)+1, max(lon_index_used_FINAL)+1,  1) 
            print(model.summary() )  
        #----------------------------------



        for  iCV in range( 5):#5-FOLD CROSS VALIDATION

            train_index=[]
            valid_index=[]
            test_index=[]
            train_year_list1CV = copy.deepcopy(index_all_list) 
        
            if  iCV==4: # there are 3 years for test in the 5th CV 
                ntestyear=3
                ntrainyear=12
                nvalidyear=4
                del train_year_list1CV[16: ]
                for i in range(4):
                    valid_index.append(shuffled_index[train_year_list1CV[i]] )        
                for i in range(4,16):
                    train_index.append(shuffled_index[train_year_list1CV[i]] )
                for i in range(ntestyear):
                    test_index.append(shuffled_index[(iCV)*4+i] )

            else:  # there are 4 years for test in the 1st to 4th CV
                ntestyear=4
                ntrainyear=12
                nvalidyear=3
                
                del train_year_list1CV[(iCV)*ntestyear:(iCV+1)*ntestyear ]
                for i in range(3):
                    valid_index.append(shuffled_index[train_year_list1CV[i]] )        
                for i in range(3,15):
                    train_index.append(shuffled_index[train_year_list1CV[i]] )
                for i in range(ntestyear):
                    test_index.append(shuffled_index[(iCV)*ntestyear+i] )

            print( train_index,valid_index, test_index)




            # -------------------------------------------
            #PREPARE TRAIN, VALID, TEST DATASET FOR OBS AND FCST ,from line 233 to line 365
            #%% select train and test dataset for obs
            obs_train_leadtime_18year = np.reshape( np.array(obs_used[:, train_index,  : ]), 
            (ntrainyear*ndaysummer*ngrid_used, 1) )
            #% validation
            obs_valid_leadtime_18year = np.reshape( np.array(obs_used[:, valid_index,  : ]), 
            (nvalidyear*ndaysummer*ngrid_used, 1) )
            #% test
            obs_test_leadtime_18year = np.reshape( np.array(obs_used[:, test_index,  : ]), 
            (ntestyear*ndaysummer*ngrid_used, 1) )


            #%% prepare embedding data for lat, lon
            nday_train = ntrainyear*ndaysummer
            lat_vec_train = tf.tile(lat_index_used_FINAL, [nday_train ])#repeat nday_train
            lon_vec_train = tf.tile(lon_index_used_FINAL, [nday_train ])
            nday_valid = nvalidyear*ndaysummer
            lat_vec_valid = tf.tile(lat_index_used_FINAL, [nday_valid ])
            lon_vec_valid = tf.tile(lon_index_used_FINAL, [nday_valid ])
            nday_test = ntestyear*ndaysummer
            lat_vec_test = tf.tile(lat_index_used_FINAL, [nday_test ])
            lon_vec_test = tf.tile(lon_index_used_FINAL, [nday_test ])


            #%% prepare DEM TRAIN AND TEST dataset
            ntrainday = ntrainyear*ndaysummer
            nvalidday = nvalidyear*ndaysummer
            ntestday = ntestyear*ndaysummer            
            #Normalize DEM to 0-1
            DEM_min = np.min(DEM_neighbor0p25_mat)
            DEM_max = np.max(DEM_neighbor0p25_mat)
            DEM_neighbor0p25_mat = (DEM_neighbor0p25_mat - DEM_min)/(DEM_max - DEM_min)
            DEM_neighbor0p25_train = np.zeros((ntrainday*ngrid_used , nlat_win, nlon_win ) , dtype='float64')
            for iday in range(ntrainday):
                DEM_neighbor0p25_train[iday*ngrid_used:( (iday+1)*ngrid_used ), :, : ] = DEM_neighbor0p25_mat
            DEM_neighbor0p25_valid = np.zeros((nvalidday*ngrid_used , nlat_win, nlon_win ), dtype='float64')
            for iday in range(nvalidday):
                DEM_neighbor0p25_valid[iday*ngrid_used:( (iday+1)*ngrid_used ), :, : ] = DEM_neighbor0p25_mat
            DEM_neighbor0p25_test = np.zeros((ntestday*ngrid_used , nlat_win, nlon_win ), dtype='float64')
            for iday in range(ntestday):
                DEM_neighbor0p25_test[iday*ngrid_used:( (iday+1)*ngrid_used ), :, : ] = DEM_neighbor0p25_mat       
            DEM_neighbor0p25_train = tf.expand_dims(DEM_neighbor0p25_train,-1)
            DEM_neighbor0p25_valid = tf.expand_dims(DEM_neighbor0p25_valid,-1)
            DEM_neighbor0p25_test = tf.expand_dims(DEM_neighbor0p25_test,-1)
            


            # %% For surface fcsts
            fcst_train_leadtime_18year = np.reshape( np.array(fcst_surf_array[:, train_index, :, 
            (4-win_half):(4+win_half+1), (4-win_half):(4+win_half+1),  :]), 
            (ntrainyear*ndaysummer*ngrid_used, nlat_win, nlon_win, n_var_SURF ) )
            # 20year * 35 day * N grid , 3 * 3 window
            fcst_min = np.min(fcst_train_leadtime_18year, axis=(0,1,2) )
            fcst_max = np.max(fcst_train_leadtime_18year, axis=(0,1,2) )
            for ivar in range(n_var_SURF):
                fcst_train_leadtime_18year[:,:,:,ivar] = (fcst_train_leadtime_18year[:,:,:,ivar]-fcst_min[ivar])/(fcst_max[ivar] - fcst_min[ivar])
            # print(pearsonr( fcst_train_leadtime_18year[:, 0, 0, 0], obs_train_leadtime_18year[:, 0]) )
            fcst_valid_leadtime_18year = np.reshape( np.array(fcst_surf_array[ :,valid_index, :,  
            (4-win_half):(4+win_half+1), (4-win_half):(4+win_half+1),  :]), 
            (nvalidyear*ndaysummer*ngrid_used, nlat_win, nlon_win, n_var_SURF ) )
            for ivar in range(n_var_SURF):
                fcst_valid_leadtime_18year[:,:,:,ivar] = (fcst_valid_leadtime_18year[:,:,:,ivar]-fcst_min[ivar])/(fcst_max[ivar] - fcst_min[ivar])
            print(pearsonr( fcst_valid_leadtime_18year[:,0, 0, 0], obs_valid_leadtime_18year[:, 0]) )
            fcst_test_leadtime_18year = np.reshape( np.array(fcst_surf_array[ :,test_index,  :, 
            (4-win_half):(4+win_half+1), (4-win_half):(4+win_half+1),  :]), 
            (ntestyear*ndaysummer*ngrid_used, nlat_win, nlon_win, n_var_SURF ) )
            for ivar in range(n_var_SURF):
                fcst_test_leadtime_18year[:,:,:,ivar] = (fcst_test_leadtime_18year[:,:,:,ivar]-fcst_min[ivar])/(fcst_max[ivar] - fcst_min[ivar])
            # print(pearsonr( fcst_test_leadtime_18year[:,0, 0, 0], obs_test_leadtime_18year[:, 0]) )


            #for pressure level dataset
            fcstPL_train_leadtime_18year = np.reshape( np.array(fcst_PL_array[:, train_index, :,  
            (4-win_half):(4+win_half+1), (4-win_half):(4+win_half+1), varlist_used]), 
            (ntrainyear*ndaysummer*ngrid_used, nlat_win, nlon_win, n_var_PL ) )
            # 20year * 35 day * N grid , 3 * 3 window
            fcstPL_min = np.min(fcstPL_train_leadtime_18year, axis=(0,1,2) )
            fcstPL_max = np.max(fcstPL_train_leadtime_18year, axis=(0,1,2) )
            for ivar in range(n_var_PL):
                fcstPL_train_leadtime_18year[:,:,:,ivar] = (fcstPL_train_leadtime_18year[:,:,:,ivar]-fcstPL_min[ivar])/(fcstPL_max[ivar] - fcstPL_min[ivar])
            # print(pearsonr( fcstPL_train_leadtime_18year[:, 0, 0, 0], obs_train_leadtime_18year[:, 0]) )
            fcstPL_valid_leadtime_18year = np.reshape( np.array(fcst_PL_array[ :,valid_index, :, 
            (4-win_half):(4+win_half+1), (4-win_half):(4+win_half+1), varlist_used]), 
            (nvalidyear*ndaysummer*ngrid_used, nlat_win, nlon_win, n_var_PL ) )
            for ivar in range(n_var_PL):
                fcstPL_valid_leadtime_18year[:,:,:,ivar] = (fcstPL_valid_leadtime_18year[:,:,:,ivar]-fcstPL_min[ivar])/(fcstPL_max[ivar] - fcstPL_min[ivar])
            # print(pearsonr( fcstPL_valid_leadtime_18year[:,0, 0, 0], obs_valid_leadtime_18year[:, 0]) )
            fcstPL_test_leadtime_18year = np.reshape( np.array(fcst_PL_array[ :,test_index,  :,
            (4-win_half):(4+win_half+1), (4-win_half):(4+win_half+1), varlist_used]), 
            (ntestyear*ndaysummer*ngrid_used, nlat_win, nlon_win, n_var_PL ) )
            for ivar in range(n_var_PL):
                fcstPL_test_leadtime_18year[:,:,:,ivar] = (fcstPL_test_leadtime_18year[:,:,:,ivar]-fcstPL_min[ivar])/(fcstPL_max[ivar] - fcstPL_min[ivar])

            fcst_PL_array.close()
            fcst_surf_array.close()
            gc.collect()



            #%% COMBINE all variables together
            if flag_DEM: #use DEM
                precip_1lead_traindata = tf.data.Dataset.from_tensor_slices( (lat_vec_train,lon_vec_train, 
                    tf.concat((  fcst_train_leadtime_18year, fcstPL_train_leadtime_18year, DEM_neighbor0p25_train),axis=-1), 
                    obs_train_leadtime_18year)  )

                precip_1lead_validdata = tf.data.Dataset.from_tensor_slices( (lat_vec_valid,lon_vec_valid, 
                    tf.concat(( fcst_valid_leadtime_18year, fcstPL_valid_leadtime_18year, DEM_neighbor0p25_valid),axis=-1), 
                    obs_valid_leadtime_18year ) )

                precip_1lead_testdata = tf.data.Dataset.from_tensor_slices( ( lat_vec_test,lon_vec_test, 
                    tf.concat(( fcst_test_leadtime_18year, fcstPL_test_leadtime_18year, DEM_neighbor0p25_test),axis=-1), 
                    obs_test_leadtime_18year ) )
                precip_1lead_testdata   
    
            else:#NO DEM
                precip_1lead_traindata = tf.data.Dataset.from_tensor_slices( (lat_vec_train,lon_vec_train, 
                    tf.concat(( fcst_train_leadtime_18year,fcstPL_train_leadtime_18year),axis=-1), 
                    obs_train_leadtime_18year)  )

                precip_1lead_validdata = tf.data.Dataset.from_tensor_slices( (lat_vec_valid,lon_vec_valid, 
                    tf.concat(( fcst_valid_leadtime_18year,fcstPL_valid_leadtime_18year),axis=-1), 
                    obs_valid_leadtime_18year ) )

                precip_1lead_testdata = tf.data.Dataset.from_tensor_slices( ( lat_vec_test,lon_vec_test, 
                    tf.concat(( fcst_test_leadtime_18year, fcstPL_test_leadtime_18year),axis=-1), 
                    obs_test_leadtime_18year ) )
                precip_1lead_testdata






            #--------------------------------------------------
            #shuffle and batch setting
            batchsize= 64 #
            batch_valid_size= 10000 #
            buffersize= obs_train_leadtime_18year.shape[0]
            # buffersize=batchsize*5# obs_train_leadtime_18year.shape[0]
            # batch_valid_size= obs_valid_leadtime_18year.shape[0] #

            train_dataset = precip_1lead_traindata.shuffle(buffersize)
            train_dataset = train_dataset.batch(batchsize)
            train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

            valid_dataset = precip_1lead_validdata.batch(batch_size=batch_valid_size)
            valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

            test_dataset = precip_1lead_testdata.batch(batch_size=batch_valid_size)
            test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)


            #setting:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            valid_loss = tf.keras.metrics.Mean(name='valid_loss')


            #% batch train 
            @tf.function
            def train_step( lat_part,lon_part, images, labels):
                with tf.GradientTape() as tape:
                    predictions = model( [lat_part,lon_part,images])
                    predict_shift = - tf.sqrt((tf.square(predictions[:,  0]) ) +1e-6)
                    predict_mu =  (tf.exp(predictions[:, 1]) )
                    predict_sigma = tf.sqrt( (tf.exp(predictions[:,  2]) ) +1e-6)
                    predict_mat = tf.stack([predict_shift,predict_mu, predict_sigma], axis=-1)
                    y_true = tf.cast(labels[:,0], dtype='float32')
                    loss = loss_CRPS_CSGD_object_V2(y_true, predict_mat)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss(loss) #update use gradient

            @tf.function
            def valid_step( lat_part,lon_part,  images, labels):
                predictions = model( [lat_part,lon_part,images])
                y_true = tf.cast(labels[:,0], dtype='float32')
                predict_shift = - tf.sqrt((tf.square(predictions[:,  0]) ) +1e-6)
                predict_mu =  (tf.exp(predictions[:, 1]) )
                predict_sigma =  tf.sqrt( (tf.exp(predictions[:,  2]) ) +1e-6)
                predict_mat = tf.stack([predict_shift,predict_mu, predict_sigma], axis=-1)
                t_loss = loss_CRPS_CSGD_object_V2(y_true, predict_mat)
                valid_loss(t_loss)






            # %-----------------------------------------
            # TRAINING 
            train_loss_vec = np.zeros((EPOCHS,))
            valid_loss_vec = np.zeros((EPOCHS,))
            n_noimprove = 0
            n_max_patience_set=10
            min_delta_set= -0.025
            valid_loss_best = 100


            for epoch in range(EPOCHS):

                #reset states of loss 
                train_loss.reset_states()
                valid_loss.reset_states()

                for   lat_part, lon_part, images, labels in train_dataset:
                    train_step( lat_part, lon_part,images, labels) # train step

                for    lat_part, lon_part, valid_images, valid_labels in valid_dataset:
                    valid_step(  lat_part, lon_part, valid_images, valid_labels)

                template = 'Epoch {}, Loss: {}, Test Loss: {}'   
                print(template.format(epoch + 1,
                                    train_loss.result()*10,
                                    valid_loss.result()*10,
                                    ))
                train_loss_vec[epoch] = np.round(train_loss.result()*1,2)
                valid_loss_vec[epoch] = np.round(valid_loss.result()*1,2)

                #early stop:
                if (valid_loss_vec[epoch] - valid_loss_best ) > min_delta_set:# if loss doesn't improve much
                    n_noimprove += 1
                else:
                    valid_loss_best = valid_loss_vec[epoch]
                    n_noimprove=0
                if (n_noimprove==n_max_patience_set):#if reach max waiting time
                    print('..............................')
                    print('Early stop at epoch ', str(epoch) )
                    break
            print('train \n',train_loss_vec)
            print('test \n',valid_loss_vec)

            #%% plot loss curve
            # import matplotlib.pyplot as plt
            # plt.plot(train_loss_vec)
            # plt.plot(valid_loss_vec)
            # plt.legend(['train loss', 'test loss'])
            # plt.title(methodname)
            # plt.savefig(file_path+'loss_curve2/'+ methodname+'-CV' + str(iCV) + '.jpg' )
        
            #%% SAVE MODEL
            file_model_path = file_path+'fittedmodel/'+ methodname +'-CV' + str(iCV)
            models.save_model(model, file_model_path)






            #TEST dataset ---------------------------------------------------
            all_valid_size = obs_test_leadtime_18year.shape[0] # here use all samples in one batch?
            batch_valid_size = all_valid_size #all days, i.e., ngrid_used * ntestyear *5
            test_dataset = precip_1lead_testdata.batch(batch_size=batch_valid_size)
            test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
            #if use quantiles as ens fcst
            n_ens =100
            bin_quantile = 1/n_ens
            quantile_vec = np.linspace( bin_quantile/2, 1-bin_quantile/2, n_ens)
            #batch prediction
            batch_prediction = np.zeros(( batch_valid_size, n_ens))
            #all days prediction
            predictpar_all_mat = np.zeros(( all_valid_size, 3))
            predictmean_all_mat = np.zeros(( all_valid_size ))
            predictens_all_mat = np.zeros(( all_valid_size,n_ens ))
            rawfcst_all = np.zeros(( all_valid_size ))
            y_true_all= np.zeros(( all_valid_size ))
            ibatch=0#initialize
            for  lat_part,lon_part,test_images, test_labels in test_dataset:

                dims_out = test_labels.shape
                predictions = model(  [lat_part,lon_part,test_images])
                predictions = predictions.numpy()
                predict_shift = -np.sqrt((np.square(predictions[:, 0]) )+1e-6)
                predict_mu = np.exp(predictions[:, 1])
                predict_sigma = np.sqrt( np.exp(predictions[:, 2])+1e-6)

                par_shape = (predict_mu/predict_sigma)**2
                par_scale = (predict_sigma**2)/predict_mu
                par_rate = 1/par_scale
                par_shift = np.array(predict_shift)
                par_shape = np.array(par_shape)
                par_scale = np.array(par_scale)

                #get quantiles from CSG distribution as the ens fcst
                for iday in range(batch_valid_size):
                    batch_prediction[iday, :] = gamma.ppf(quantile_vec, par_shape[iday] , par_shift[iday], par_scale[iday] )
                batch_prediction[batch_prediction<0]=0#neg to 0
                test_ENSMEAN_images_oriscale = np.mean(batch_prediction,axis=-1)

                #raw fcst
                rawfcst_oriscale = test_images[:, win_half, win_half,  0] 
                y_true = test_labels[:,0] 
                predict_par_mat_batch = np.stack([predict_shift,predict_mu, predict_sigma], axis=-1) 

                #save predicted par
                predictpar_all_mat[ibatch*batch_valid_size:(ibatch+1)*batch_valid_size, :] = predict_par_mat_batch
                #save raw fcst and obs
                rawfcst_all [ibatch*batch_valid_size:(ibatch+1)*batch_valid_size] = rawfcst_oriscale  
                y_true_all [ibatch*batch_valid_size:(ibatch+1)*batch_valid_size ] = y_true  
                predictmean_all_mat[ibatch*batch_valid_size:(ibatch+1)*batch_valid_size ] = test_ENSMEAN_images_oriscale      
                predictens_all_mat[ibatch*batch_valid_size:(ibatch+1)*batch_valid_size,: ] = batch_prediction      
            
                #verification
                print(pearsonr(rawfcst_oriscale , y_true  ))
                print(pearsonr(test_ENSMEAN_images_oriscale , y_true  ))

                ibatch=ibatch+1



            # reshape to nday * nyear * ngrid
            ngrid_nyear = ngrid_used*ntestyear
            for iday in range(35):
                for iyear in range(ntestyear):
                    predictens_array[iday, test_index[iyear], :, : ] = \
                    predictens_all_mat[(iday*ngrid_nyear+iyear*ngrid_used):( (iday)*ngrid_nyear+(iyear+1)*ngrid_used), : ]
                    predictmean_array[iday, test_index[iyear], : ] = \
                    predictmean_all_mat[(iday*ngrid_nyear+iyear*ngrid_used):( (iday)*ngrid_nyear+(iyear+1)*ngrid_used) ]
    







        #SAVE post-processed results---------------------------
        predictens_array = xr.DataArray(predictens_array,  
            coords={
                'day':np.arange(ndaysummer),
                'year':np.arange(19),
                'grid':np.arange(ngrid_used),
                'member':np.arange(100),
            },
            dims=( 'day','year', 'grid', 'member'),
        )

        predictens_array.to_netcdf(file_path+'output3/CNNfcstENS'+ methodname +  '_earlystop_5CV.nc') 


        predictmean_array = xr.DataArray(predictmean_array,  
            coords={
                'day':np.arange(ndaysummer),
                'year':np.arange(19),
                'grid':np.arange(ngrid_used),
            },
            dims=( 'day','year', 'grid'),
        )

        predictmean_array.to_netcdf(file_path+'output3/CNNfcst'+ methodname +  '_earlystop_5CV.nc') 


