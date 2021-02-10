import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
BATCH_SIZE = 256

def load_data():
    # Default values.
    train_set = 'data/UNSW_NB15_training-set.csv'
    test_set = 'data/UNSW_NB15_testing-set.csv'
    # train_set = 'data/UNSW_NB15_training-set_multi.csv'
    # test_set = 'data/UNSW_NB15_testing-set_multi.csv'
    train = pd.read_csv(train_set, index_col='id')
    test = pd.read_csv(test_set, index_col='id')

    # 二分类数据
    training_label = train['label'].values
    testing_label = test['label'].values


    train.drop(['label', 'attack_cat'], axis=1, inplace=True)
    # train.drop(['label', 'attack_cat','ct_srv_src', 'ct_dst_ltm','ct_src_dport_ltm',
    #            'ct_dst_sport_ltm', 'ct_dst_src_ltm'], axis=1, inplace=True)
    test.drop(['label', 'attack_cat'], axis=1, inplace=True)
    # test.drop(['label', 'attack_cat','ct_srv_src', 'ct_dst_ltm','ct_src_dport_ltm',
    #            'ct_dst_sport_ltm', 'ct_dst_src_ltm'], axis=1, inplace=True)
    train, temp_train = df_to_dataset(train, training_label, batch_size=BATCH_SIZE)
    test, temp_test = df_to_dataset(test, testing_label, batch_size=BATCH_SIZE)
    train_set = train
    test_set = test

    return train_set, temp_train, test_set, temp_test

def  df_to_dataset(dataframe, labels,batch_size=BATCH_SIZE):
    dataframe = dataframe.values
    sample = dataframe.shape[0]
    features = dataframe.shape[1]
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataframe = scaler.fit_transform(dataframe)
    dataframe = np.array(dataframe)
    labels = np.array(labels)
    new_labels = np.reshape(labels,(sample,1))
    new_dataframe = np.reshape(dataframe, (sample, features, 1))
    print('------------------------------------------')
    print(new_dataframe.shape)
    print('------------------------------------------')
    print(new_labels.shape)
    return new_dataframe,new_labels
