import os
import torch


class Config():
    """
     Transformer Translation
    """

    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        # TODO: dataset_dir should be an input param
        self.dataset_dir = os.path.join(self.project_dir, 'data/trace/')
        self.train_corpus_file_paths = os.path.join(self.dataset_dir, 'train.json')
        self.test_corpus_file_paths = os.path.join(self.dataset_dir, 'test.json')
        self.model_name = '0_hop_trace.pt'
        self.load_graph = True
        self.min_freq = 1
        self.max_sen_len = None

        self.batch_size = 1
        # TODO: vocab_size should be an input param, depends on the number of value in cadet_tree.txt
        self.vocab_size = 162
        self.node2vec_size = 32  # srcUUID / dstUUID
        self.d_model = 64
        self.num_head = 8
        self.num_encoder_layers = 1
        self.num_decoder_layers = 0
        self.dim_feedforward = 64
        self.dim_classification = 32
        self.num_class = 2
        self.dropout = 0.1
        self.concat_type = 'sum'
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 10e-9
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = 2000
        self.model_save_dir = os.path.join(self.project_dir, 'saved_model')
        self.model_save_per_epoch = 1
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)


if __name__ == '__main__':
    config = Config()
    print(config.project_dir)
    print(config.train_corpus_file_paths)
