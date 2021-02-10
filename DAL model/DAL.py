import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from data_processing import load_data
from DenseLayer import DenseBlock,TransitionLayer
import  sklearn as sk
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU,AlphaDropout,BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
from keras.regularizers import l2 
from keras import backend as K
from  AttentionLayer import AttentionLayer
from tensorflow.keras.models import load_model

from openpyxl import workbook                       # 写Excel
from openpyxl import load_workbook                  # 读Excel

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

BATCH_SIZE = 256

def plt_show(history):
    history_dict = history.history
    history_dict.keys()
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    # "bo" is for "blue dot"
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    # plt.clf()  # clear figure
    plt.plot(epochs, acc, 'b-', label='UNSW-NB15_training-set')
    plt.plot(epochs, val_acc, 'r', label='UNSW-NB15_testing-set')
    # plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc=4)
    plt.show()



print("Load data...")
train, train_label, test, test_label = load_data()  # data:(, 196) label:(, 10)
print("train_label finish: ", train_label.shape)

def Swish(x):
    return x*K.sigmoid(0.1*x)

SINGLE_ATTENTION_VECTOR = False
def attention_block(inputs):
    print(inputs.shape)
    input_dim = int(inputs.shape[2])
    a = layers.Permute((2, 1))(inputs)
    a = layers.Dense(1, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = layers.Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = layers.RepeatVector(input_dim)(a)
    a_probs = layers.Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = layers.multiply([inputs, a_probs], name="attention_mul")
    return output_attention_mul



inputs = tf.keras.Input(shape=(42, 1))
# print(inputs)
#dilation conv
# # x = layers.Conv1D(10, 3, activation='tanh')(inputs)
# x = layers.Conv1D(10, 3, activation = 'tanh',dilation_rate = 1)(inputs)
# #x = LeakyReLU(alpha=0.5)(x)
# x = BatchNormalization(axis = 1)(x)
# # x = layers.MaxPool1D(pool_size=3)(x)
# x = layers.Conv1D(20, 3,activation = 'elu',dilation_rate = 3)(x)
# x = BatchNormalization(axis = 1)(x)
# # x = LeakyReLU(alpha=0.3)(x)
# # x = layers.Conv1D(16, 3,activation = 'elu')(x)
# x = layers.Conv1D(30, 3,activation = 'elu',dilation_rate = 7)(x)
# x = BatchNormalization(axis = 1)(x)
# # x = layers.Conv1D(10, 1, activation='elu')(x)
# x = attention_block(x)

#DenseNet
x = layers.Conv1D(60, 3, activation='tanh')(inputs)
# x = layers.AlphaDropout(0.17)(x)
x = DenseBlock(x,2,60,drop_rate=0.25, ax=1)
# x = TransitionLayer(x, ax=1)
# x = DenseBlock(x,2,60,drop_rate=0.7, ax=1)
# x = TransitionLayer(x, ax=1)
# print('2',x)
# # x = layers.MaxPool1D(pool_size=3)(x)
# x = layers.Conv1D(10, 3, activation='relu')(inputs)
# x = layers.Conv1D(20, 5, activation='relu')(x)
# x = layers.Conv1D(30, 6, activation='relu')(x)
# x = layers.Dense(20, activation='elu')(x)
# x = layers.Bidirectional(layers.LSTM(420, activation='elu', recurrent_activation='hard_sigmoid',
#                                        return_sequences=False))(x)
# x = layers.CuDNNLSTM(420)(x)
# x = LeakyReLU(alpha=0.3)(x)
x = attention_block(x)
# x = TransitionLayer(x, ax=1)
# print('3',x)
# x = layers.AlphaDropout(0.3)(x)
# x = layers.Bidirectional(layers.LSTM(400, activation='elu', recurrent_activation='hard_sigmoid',kernel_regularizer=l2(0.3),
#                                         return_sequences=False))(x)
# x = layers.Bidirectional(layers.LSTM(120, activation='elu', recurrent_activation='hard_sigmoid',
#                                        return_sequences=False))(x)
# x = layers.AlphaDropout(0.17)(x)
x = layers.CuDNNLSTM(60,return_sequences=True)(x)
x = LeakyReLU(alpha=0.3)(x)
# x = BatchNormalization(axis=1)(x)
x = layers.AlphaDropout(0.17)(x)
x = TransitionLayer(x, ax=1)
# x = layers.CuDNNLSTM(512,kernel_regularizer=l2(0.05))(x)
# x = layers.Flatten()(x)
# x = layers.AlphaDropout(0.2)(x)
# x = layers.LSTM(80, activation='relu', return_sequences=False)(x)
# x = layers.Dropout(0.2)(x)
# x = layers.Dense(256,activation = 'elu',kernel_regularizer=l2(0.05))(x)
# x = layers.Dense(90,activation = 'elu')(x)
# x = layers.AlphaDropout(0.17)(x)
# x = layers.Dense(30,activation = 'elu')(x)
# x = LeakyReLU(alpha=0.2)(x)
# x = layers.AlphaDropout(0.17)(x)
x = layers.Dense(30,activation = 'elu')(x)
# x = BatchNormalization(axis=1)(x)
x = layers.AlphaDropout(0.17)(x)
# x = layers.Flatten()(x)
# x = layers.Dense(84,activation = 'elu',kernel_regularizer=l2(0.05))(x)
# outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
# print('4',x)
# x = layers.Lambda(lambda x:K.mean(x,axis=0,keepdims=True),name='reshapelayer')(x)
# print('5',x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "history_training/best_model-{epoch:02d}-{acc:.4f}-{val_acc:.4f}.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_acc',verbose=1, save_best_only=True,mode='auto',save_weights_only = True, period=1)
tbCallBack = TensorBoard(log_dir="./model",histogram_freq=0, write_graph=True, write_grads=True,
                         write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
callbacks_list = [cp_callback]


adam = Adam(lr=0.001)
# adam = Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# train_step_MomentumOptimizer = tf.train.MomentumOptimizer(0.0005, 0.9)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train, train_label, batch_size=1024, epochs=120, validation_data=(test, test_label),shuffle=True, verbose=1,callbacks = callbacks_list)

# print('--------------score--------------')
# score = model.evaluate(test, test_label, verbose=1)
# print("loss:",score[0])
# print("accu:",score[1])
# writer=tf.summary.FileWriter("Neural_Networks/",tf.get_default_graph())
# writer.close()
# model_best = load_model("history_training/best_model.hdf5")
# print('--------------score--------------')
# score = model_best.evaluate(test, test_label, verbose=1)
# print("loss:",score[0])
# print("accu:",score[1])
# y_pred = model_best.predict(test)
# y_pred = np.argmax(y_pred, axis = 1)
# print('pred')
# print(y_pred)
# print('true')
# print(test_label)
# y_true = np.argmax(test, axis = 1)
# matrix = confusion_matrix(test_label,y_pred)
# print('Testset Confusion Matrix')
# print(matrix)
# print('Classification Report')
# print(classification_report(test_label,y_pred))

plt_show(history)

# model.summary()
