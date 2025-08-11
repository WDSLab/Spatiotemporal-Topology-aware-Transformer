
import numpy as np
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

def data_provider(root_path, batch_size, seq_len, pred_len, label_len=0, flag='train', drop_last=False, print_=True, future=False):
    
    data = np.load(root_path)
    date_index = pd.date_range(start="2013-01-01", end="2022-12-31", freq="D")
    date_index = pd.DataFrame({"date": date_index})

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = batch_size  
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = batch_size  

    if future:
        data_set = Dataset_ASTD_future(
            df=data,
            date=date_index,
            flag=flag,
            size=[seq_len, label_len, pred_len]
        )
    else:
        data_set = Dataset_ASTD(
        df=data,
        date=date_index,
        flag=flag,
        size=[seq_len, label_len, pred_len]
        )
    if print_ == True:
        print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last)
    
    return data_set, data_loader
            
class Dataset_ASTD(Dataset):
    def __init__(self, df, date, scale=False, flag='train', size=None, data_path=None):

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.df_raw = df
        self.scale = scale
        self.date = date

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()

        total_seq_len = self.seq_len + self.pred_len
        total_windowable_len = len(self.df_raw) - total_seq_len + 1  
        sample_indices = list(range(total_windowable_len))
        num_train = int(len(sample_indices) * 0.7)
        num_val = int(len(sample_indices) * 0.1)
        num_test = len(sample_indices) - num_train - num_val

        train_indices = sample_indices[:num_train]
        val_indices = sample_indices[num_train:num_train+num_val]
        test_indices = sample_indices[num_train+num_val:]

        # num_train = int(len(self.df_raw) * 0.8)   # test only
        # num_test = int(len(self.df_raw) * 0.2)

        if self.set_type == 0:
            self.indices = train_indices
        elif self.set_type == 1:
            self.indices = val_indices
        else:
            self.indices = test_indices

        # border1s = [0, 0, num_train - self.pred_len]  # test only
        # border2s = [num_train, 0, len(self.df_raw)]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]

        df_data = self.df_raw

        if self.scale:
            df_data = df_data.reshape(B*W*H, F)
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        df_stamp = self.date[['date']]
        # df_stamp = self.date[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp = df_stamp.drop(['date'], 1).values

        self.data_x = data
        self.data_y = data
        # self.data_x = data[border1:border2]
        # self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = self.indices[index]
        # s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_x_mark, seq_y, seq_y_mark

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.indices)
