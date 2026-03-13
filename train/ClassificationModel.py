import torch
import torch.nn as nn
from MyTransformer import MyTransformerEncoder, MyTransformerEncoderLayer
from Embedding import PositionalEncoding, TokenEmbedding
import random
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

seed = 100
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

class ClassificationModel(nn.Module):
    def __init__(self,
                 load_graph=True,
                 vocab_size=630,
                 node_vec_size=128,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=2048,
                 dim_classification=64,
                 num_classification=4,
                 dropout=0.1):
        super(ClassificationModel, self).__init__()
        self.load_graph = load_graph
        pos_size = d_model+node_vec_size
        self.pos_embedding = PositionalEncoding(d_model=d_model, dropout=dropout)
        # TODO: decrease Token_Embed dim, token_dim + node2vec_dim = pos_dim
        self.src_token_embedding = TokenEmbedding(vocab_size, d_model)

        encoder_layer = MyTransformerEncoderLayer(d_model, nhead,
                                                  dim_feedforward,
                                                  dropout)
        model_encoder_norm = nn.LayerNorm(d_model)
        # self.encoder_norm = nn.LayerNorm(d_model)
        # self.node2vec_norm = nn.LayerNorm(node_vec_size)

        self.encoder = MyTransformerEncoder(encoder_layer,
                                            num_encoder_layers, model_encoder_norm)
        self.classifier = nn.Sequential(nn.Linear(pos_size, dim_classification),
                                        nn.Dropout(dropout),
                                        nn.Linear(dim_classification, num_classification))

    def forward(self,
                src,  # [src_len, batch_size]
                node_vec,
                src_mask=None,
                src_key_padding_mask=None,  # [batsh_size, src_len]
                concat_type='sum'
                ):
        src_embed = self.src_token_embedding(src)  # [src_len, batch_size, embed_dim] [316, 2, 512]

        # token_dim + node2vec_dim = pos_dim
        src_embed = self.pos_embedding(src_embed)  # [src_len, batch_size, embed_dim + node2vec_dim]
        memory = self.encoder(src=src_embed,
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)  # [src_len,batch_size,embed_dim]

        if concat_type == 'sum':
            memory = torch.sum(memory, dim=0)
        elif concat_type == 'avg':
            memory = torch.sum(memory, dim=0) / memory.size(0)
        else:
            memory = memory[-1, ::]

        # append the z_embed here
        if self.load_graph:
            z_embed = torch.transpose(node_vec, 0, 1)
            z_embed = nn.functional.normalize(z_embed, p=2.0, dim=1)
            graph_memory = torch.cat( (memory ,z_embed) ,-1 )
            # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
            out = self.classifier(graph_memory)  # logits
        else:
            out = self.classifier( memory )
        return out  # [batch_size, num_class]


if __name__ == '__main__':
    src_len = 7
    batch_size = 2
    dmodel = 32
    num_head = 4
    src = torch.tensor([[4, 3, 2, 6, 0, 0, 0],
                        [5, 7, 8, 2, 4, 0, 0]]).transpose(0, 1)  # [src_len, batch_size]
    src_key_padding_mask = torch.tensor([[True, True, True, True, False, False, False],
                                         [True, True, True, True, True, False, False]])
    model = ClassificationModel(vocab_size=10, d_model=dmodel, nhead=num_head)
    logits = model(src, src_key_padding_mask=src_key_padding_mask)
    print(logits.shape)
