import random
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import json
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

seed = 100
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
     List padding
        sequences:
        batch_first: batch_size
        padding_value:
        max_len : None batch padding
         max_len batch batch padding
         max_len
    Returns:
    """
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    length = max_len
    max_len = max([s.size(0) for s in sequences])
    if length is not None:
        max_len = max(length, max_len)
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor


class LoadSentenceClassificationDataset():
    def __init__(self,
                 load_graph=True,
                 batch_size=20,
                 min_freq=1,  # min_freq
                 max_sen_len='same'):
        # max_sen_len = None batch padding
        self.min_freq = min_freq
        self.specials = ['<unk>', '<pad>']
        self.PAD_IDX = 0
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        self.load_graph = load_graph

    def data_process(self, semi_data_path):
        """
        
        :param filepath:
        :return:
        """
        print( "Loading resampled datasets ..." )
        resampling_in = open(semi_data_path)
        # x: sequence
        # y: label
        # z: node2vec array for each src id, dst id
        x_y_list = json.load( resampling_in )
        resampling_in.close()
        x_train = x_y_list[0]
        y_train = x_y_list[1]
        z_train = x_y_list[2]
        print( "Loaded" )

        data_itr = []
        for idx ,value in enumerate( x_train ):
            if self.load_graph:
                x_tensor = torch.tensor( x_train[idx] ,dtype=torch.int32 )
                y_tensor = torch.tensor( y_train[idx] ,dtype=torch.int32 )
                z_tensor = torch.tensor( z_train[idx] ,dtype=torch.float32 )

                tmp = tuple( [x_tensor ,y_tensor, z_tensor] )
            else:
                x_tensor = torch.tensor( x_train[idx] ,dtype=torch.int32 )
                y_tensor = torch.tensor( y_train[idx] ,dtype=torch.int32 )

                tmp = tuple( [x_tensor ,y_tensor] )
            data_itr.append( tmp )

        list_len = [len( i ) for i in x_train]
        max_len = max( list_len )

        return data_itr, max_len

    def load_train_val_test_data(self, train_file_paths, test_file_paths):
        g = torch.Generator()
        g.manual_seed( 0 )

        # TODO: update train_data, test_data from DARPA resampling.json
        train_data, max_sen_len = self.data_process(train_file_paths)

        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        test_data, _ = self.data_process(test_file_paths)

        # Torch reproducibility
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter

    def generate_batch(self, data_batch):
        if self.load_graph:
            batch_sentence, batch_label, batch_node_vec = [], [], []
            for (sen, label, node_vec) in data_batch:  # batch
                batch_sentence.append(sen)
                batch_label.append(label)
                batch_node_vec.append(node_vec)
            batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                          padding_value=self.PAD_IDX,
                                          batch_first=False,
                                          max_len=self.max_sen_len)
            batch_label = torch.tensor(batch_label, dtype=torch.long)

            batch_node_vec = pad_sequence(batch_node_vec,  # [batch_size,max_len]
                                          padding_value=self.PAD_IDX,
                                          batch_first=False,
                                          max_len=self.max_sen_len)

            return batch_sentence, batch_label, batch_node_vec
        else:
            batch_sentence ,batch_label ,batch_node_vec = [] ,[] ,[]
            for (sen ,label) in data_batch:  # batch
                batch_sentence.append( sen )
                batch_label.append( label )
            batch_sentence = pad_sequence( batch_sentence ,  # [batch_size,max_len]
                                           padding_value=self.PAD_IDX ,
                                           batch_first=False ,
                                           max_len=self.max_sen_len )
            batch_label = torch.tensor( batch_label ,dtype=torch.long )

            return batch_sentence ,batch_label

if __name__ == '__main__':
    path = "data/debug/test.json"
    data_loader = LoadSentenceClassificationDataset(train_file_path=path,
                                                    max_sen_len=None)
    data, max_len = data_loader.data_process(path)
    train_iter, test_iter = data_loader.load_train_val_test_data(path, path)
    for sample, label, node_vec in train_iter:
        print(sample.shape)  # [seq_len,batch_size]
