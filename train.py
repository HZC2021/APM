from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Activation, RNN
import tensorflow as tf
import os

from keras.layers import LSTM
import numpy as np
from pickle import dump
from keras.regularizers import L1
import math
from GCN_LSTM import CustomLSTMCell
# from kerasrnn import RNN
import random
import datetime
# from GCNlayer import MGCN

def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * 1 / (1 + 5e-5 * epoch)



def create_model(feature=6, units=1, drop_out=0., l1=1e-3, batch_input_shape=None):
    model = Sequential()
    # model = MGCN(units, dropout_rate = drop_out, l1_reg = l1)
    model.add(RNN(CustomLSTMCell(feature,units, dropout=drop_out, recurrent_dropout=drop_out,
                            kernel_regularizer=L1(l1=l1)),
                  unroll=True, #return_sequences=True,
                  stateful=True))
    # model.add(Activation('relu'))
    # model.add(Dense(1, kernel_regularizer=L1(l1=l1)))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss='mse')
    return model


def trainwithval(trainset, valset, units=1, drop_out=0., batch_size=32, epochs=50, L1_value=0., L1_name="0"):
    # design network

    model = create_model(units=units, drop_out=drop_out, l1=L1_value, )
    # batch_input_shape = [batch_size, trainset[0].shape[1], trainset[0].shape[2]])
    # define callback func

    file_name = "{epoch:02d}_{loss:.6f}.ckpt"
    checkpoint_path = os.path.join('./std_record/train_L%s_u%d_dp%.2f_std_t0' % ( L1_name, units, drop_out), file_name)
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callback_savemodel = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=0,
        save_weights_only=True,
        save_freq="epoch")
    # devide data for training and validation
    train_X, train_y = trainset
    val_X, val_y = valset
    # fit network
    val_epoch = []
    for i in range(epochs):
        history = model.fit(train_X, train_y, epochs=1, batch_size=batch_size,  # validation_data=(val_X, val_y),
                            callbacks=[callback_lr, callback_savemodel], verbose=2,
                            shuffle=False)
        model.reset_states()
        val_len = len(val_y)
        # train_len = len(train_y)
        # run_X = np.concatenate((train_X, val_X), axis=0)
        # run_y = np.concatenate((train_y, val_y), axis=0)
        run_yhat = model.predict(val_X, batch_size=batch_size)
        # tmp = mean_squared_error(run_yhat[-val_len:, :], val_y[:, :])
        # tmp = mean_squared_error(run_yhat[0:-val_len, :], train_y[:, :])
        tmp = mean_squared_error(run_yhat[:, :], val_y[:, :])
        print("epoch:%d val mse:" % i, tmp)
        val_epoch.append([i, tmp, history.history['loss']])
        model.reset_states()
    np.save(r"std\val_epoch_L%s_u%d_dp%.2f_t0.npy" % (L1_name, units, drop_out), val_epoch)

    # plot history
    # pyplot.plot(history.history['loss'], label='train')
    # # pyplot.plot(history.history['val_loss'], label='val')
    # pyplot.legend("Train and Test Loss from the S-LSTM model")
    # # pyplot.show()
    # pyplot.savefig('./train/SLSTM.jpg')
    return model


# # make a prediction
# def pred(testset, modelpath, units=50):
# 	# build model
# 	model = Sequential()
# 	model.add(RNN(SLSTMCell(units)))
# 	model.add(Dense(1))
# 	model.compile(loss='mse', optimizer='adam')
# 	model.load_weights(modelpath)
# 	test_X, test_y = testset
# 	yhat = model.predict(test_X)
# 	return yhat

if __name__ == "__main__":

    x = np.load("sample_data_100.npy")
    lbl = np.load("density_label_100.npy")
    a = np.load("AdjMat_100.npy")
    AX = np.load("AX_100.npy")
    ax = np.concatenate((a,x[:,:,4:]), axis = 2)

    timestep = len(AX[0])
    # X = []
    # y = []
    # for i in range(len(ax)-1):
    #     t = []
    #     for j in range(timestep):
    #         t.append(ax[i+j])
    #     X.append(t)
    #     y.append([lbl[i+timestep-1]])
    # X = np.array(X)
    # y = np.array(y)

    # X = []
    # y = []
    # for i in range(10):
    #     t = []
    #     for j in range(2):
    #         ax = []
    #         for i in range(1000):
    #             ax.append([1.0]*1008)
    #         t.append(ax)
    #     y.append([1.0])
    #     X.append(t)
    # X = np.array(X)
    # y = np.array(y)
    Total = len(AX)
    train_num = round(Total*0.7)
    val_num = round(Total * 0.2)
    test_num = Total - train_num - val_num
    # val_num = 2
    X = np.delete(AX, [0,1,2,3,8,9], 2)
    y = np.array([[lbl[i]] for i in range(len(lbl))])
    train_X = X[:train_num]
    train_y = y[:train_num]
    im = np.mean(train_X,axis=(0,1))
    istd = np.std(train_X,axis=(0,1))
    val_X = X[train_num:train_num + val_num]
    val_y = y[train_num:train_num + val_num]
    test_X = X[train_num + val_num:Total]
    test_y = y[train_num + val_num:Total]
    for i in range(train_X.shape[0]):
        for j in range(train_X.shape[1]):
            for k in range(train_X.shape[2]):
                train_X[i,j,k] = (train_X[i,j,k] - im[k]) / istd[k]
    for i in range(val_X.shape[0]):
        for j in range(val_X.shape[1]):
            for k in range(val_X.shape[2]):
                val_X[i,j,k] = (val_X[i,j,k] - im[k]) / istd[k]
    for i in range(test_X.shape[0]):
        for j in range(test_X.shape[1]):
            for k in range(test_X.shape[2]):
                test_X[i,j,k] = (test_X[i,j,k] - im[k]) / istd[k]
    m = np.mean(train_y[:,:])
    s = np.std(train_y[:,:])
    np.save("y_mean.npy",m)
    np.save("y_std.npy",s)
    # im = np.mean(train_X[:,:,:,train_X.shape[2]:],axis = (0,1,2))
    # istd = np.std(train_X[:, :, :, train_X.shape[2]:], axis = (0,1,2))
    # for i in range(train_X.shape[0]):
    #     for j in range(train_X.shape[1]):
    #         for k in range(train_X.shape[2]):
    #             for r in range(train_X.shape[2],train_X.shape[3]):
    #                 train_X[i,j,k,r] = (train_X[i,j,k,r] - im[r-train_X.shape[2]]) / istd[r-train_X.shape[2]]
    # for i in range(val_X.shape[0]):
    #     for j in range(val_X.shape[1]):
    #         for k in range(val_X.shape[2]):
    #             for r in range(train_X.shape[2],val_X.shape[3]):
    #                 val_X[i,j,k,r] = (val_X[i,j,k,r] - im[r-train_X.shape[2]]) / istd[r-train_X.shape[2]]
    # for i in range(test_X.shape[0]):
    #     for j in range(test_X.shape[1]):
    #         for k in range(test_X.shape[2]):
    #             for r in range(train_X.shape[2],test_X.shape[3]):
    #                 test_X[i,j,k,r] = (test_X[i,j,k,r] - im[r-train_X.shape[2]]) / istd[r-train_X.shape[2]]
    # np.save("y_mean.npy",im)
    # np.save("y_std.npy",istd)

    # for i in range(len(train_y)):
    train_y = (train_y - m) / s
    # for i in range(len(val_y)):
    val_y = (val_y - m) / s
    # for i in range(len(test_y)):
    test_y = (test_y - m) / s
    np.save("train_X.npy", train_X)
    np.save("val_X.npy", val_X)
    np.save("test_X.npy", test_X)
    np.save("train_y.npy", train_y)
    np.save("val_y.npy", val_y)
    np.save("test_y.npy", test_y)
    ## standardizatrion
    # for i range(8):
    #     m = np.mean(train_X[:,:,:,1000+i:])
    #     std = np.std(train_X[:,:,:,1000+i:])
    # train_X
    # train model
    batch_size = 1
    L1_value = [1e-5]
    L1_name = ["1e-5"]
    # L1_value = [1e-5]
    # L1_name = ["1e-5"]
    for i in range(len(L1_value)):
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        print(train_X.shape)
        model = trainwithval([train_X, train_y], [val_X, val_y], units=1,
                             drop_out=0.3, batch_size=1, epochs=100, L1_value=L1_value[i], L1_name=L1_name[i])

        trainyhat = model.predict(train_X, batch_size=1)

        rmse = math.sqrt(mean_squared_error(train_y[:, :], trainyhat[:, :])) * s
        # R2 = r2_score(train_y[:, :], trainyhat[:, :])

        print('Train RMSE: %.3f' % rmse)
        # print("Train R2 Score: %.3f" % R2)
        model.reset_states()
        testyhat = model.predict(test_X, batch_size=1)
        rmse = math.sqrt(mean_squared_error(test_y[:, :], testyhat[:, :])) * s
        print('Test RMSE: %.3f' % rmse)
        model.reset_states()
        # np.save("train_y.npy", train_y)
        # np.save("test_y.npy", test_y)



