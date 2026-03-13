import random
import torch.nn as nn
import torch
from config import Config
from data_helper import LoadSentenceClassificationDataset
from ClassificationModel import ClassificationModel
import os
import time
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

seed = 100
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

class CustomSchedule(nn.Module):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.step = 1.

    def __call__(self):
        arg1 = self.step ** -0.5
        arg2 = self.step * (self.warmup_steps ** -1.5)
        self.step += 1.
        return (self.d_model ** -0.5) * min(arg1, arg2)


def train_model(config):
    data_loader = LoadSentenceClassificationDataset(load_graph=config.load_graph,
                                                    batch_size=config.batch_size,
                                                    min_freq=config.min_freq,
                                                    max_sen_len=config.max_sen_len)
    train_iter, test_iter = data_loader.load_train_val_test_data(
        config.train_corpus_file_paths, config.test_corpus_file_paths)

    # torch.save( train_iter ,'train_data_loader.pth' )
    # torch.save( test_iter ,'test_data_loader.pth' )

    classification_model = ClassificationModel(load_graph=config.load_graph,
                                               vocab_size=config.vocab_size,
                                               node_vec_size=config.node2vec_size,
                                               d_model=config.d_model,
                                               nhead=config.num_head,
                                               num_encoder_layers=config.num_encoder_layers,
                                               dim_feedforward=config.dim_feedforward,
                                               dim_classification=config.dim_classification,
                                               num_classification=config.num_class,
                                               dropout=config.dropout)

    for p in classification_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model_save_path = os.path.join(config.model_save_dir, config.model_name)
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        classification_model.load_state_dict(loaded_paras)
        print("## Successfully loaded existing model, resuming training...")
    classification_model = classification_model.to(config.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    # learning_rate = CustomSchedule(config.d_model)
    # optimizer = torch.optim.AdamW(classification_model.parameters(),
    #                              lr=0.,
    #                              betas=(config.beta1, config.beta2),
    #                              eps=config.epsilon)
    optimizer = torch.optim.AdamW(
        [
            {"params": classification_model.pos_embedding.parameters() ,"lr": 1e-6} ,
            {"params": classification_model.src_token_embedding.parameters(), "lr": 5e-6} ,
            {"params": classification_model.encoder.parameters() ,"lr": 1e-5} ,
            {"params": classification_model.classifier.parameters() ,"lr": 1e-4} ,
        ],
        lr = 5e-4 ,
    )

    classification_model.train()
    max_f1_score = 0
    count = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        if config.load_graph:
            for idx, (sample, label, node2vec_z) in enumerate(train_iter):
                # load samples (x_train)
                sample = sample.to(config.device)  # [src_len, batch_size]
                # load labels (y_train)
                label = label.to(config.device)
                # load node2vec vector (z_train)
                node2vec_z = node2vec_z.to(config.device)

                padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
                logits = classification_model(sample,
                                              node_vec=node2vec_z,
                                              src_key_padding_mask=padding_mask)  # [batch_size,num_class]
                optimizer.zero_grad()
                loss = loss_fn(logits, label)
                loss.backward()
                # lr = learning_rate()
                # lr = 0.0001
                # for p in optimizer.param_groups:
                #     p['lr'] = lr
                optimizer.step()
                losses += loss.item()

                acc = (logits.argmax(1) == label).float().mean()
                if idx % 10 == 0:
                    print(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                          f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        else:
            for idx, (sample, label) in enumerate(train_iter):
                # load samples (x_train)
                sample = sample.to(config.device)  # [src_len, batch_size]
                # load labels (y_train)
                label = label.to(config.device)
                # no graph data
                node2vec_z = 0

                padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
                logits = classification_model(sample,
                                              node_vec=node2vec_z,
                                              src_key_padding_mask=padding_mask)  # [batch_size,num_class]
                optimizer.zero_grad()
                loss = loss_fn(logits, label)
                loss.backward()
                # lr = learning_rate()
                # lr = 0.0001
                # for p in optimizer.param_groups:
                #     p['lr'] = lr
                optimizer.step()
                losses += loss.item()

                acc = (logits.argmax(1) == label).float().mean()
                if idx % 10 == 0:
                    print(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                          f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if epoch % config.model_save_per_epoch == 0:
            y_model, y_truth = evaluate(test_iter, classification_model, config.device)
            # start testing
            SELF_TRAINING_Threshold = [0.1, 0.2, 0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9]
            for x in SELF_TRAINING_Threshold:
                p_index = [idx for idx ,val in enumerate( y_model ) if val > x]
                y_test_by_model = np.zeros( len( y_model ) )
                y_test_by_model[p_index] = 1

                precision ,recall ,test_auc ,F1_score ,TP ,FP ,TN ,FN = perf_measure( y_truth ,
                                                                                      y_test_by_model )
                print( "=============================================================================" )
                print( "test Threshold: " ,x )
                print( "=============================================================================" )
                print( "TP: " ,TP ,"\nFP: " ,FP ,"\nTN: " ,TN ,"\nFN: " ,FN )
                print( "=============================================================================" )
                print( "Precision: " ,precision ,"\nRecall: " ,recall ,"\nAUC: " ,test_auc ,"\nF1: " ,F1_score )
                print( f"max acc on test {max_f1_score:.3f}" )

                if F1_score > max_f1_score:
                    count += 1
                    max_f1_score = F1_score
                    state_dict = classification_model.state_dict()
                    # model_save_path = model_save_path + '_' + str(count)
                    torch.save( state_dict, model_save_path )

def flatten(xss):
    '''
    Flatten a list of sub lists
    e.g.:
    Input:  [[20180406113149], [20180406120452]]
    Output: [20180406113149, 20180406120452]
    '''
    return [x for xs in xss for x in xs]

def perf_measure(y_truth ,y_model):
    # TODO: how to calculate when TP=0?
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range( len( y_model ) ):
        if y_truth[i] == y_model[i] == 1:
            TP += 1
        if y_model[i] == 1 and y_truth[i] != y_model[i]:
            FP += 1
        if y_truth[i] == y_model[i] == 0:
            TN += 1
        if y_model[i] == 0 and y_truth[i] != y_model[i]:
            FN += 1

    if TP == 0:
        #raise ValueError('TP=0, the model testing accuracy is too bad!!!')
        precision, recall, F1_score = 0, 0, 0
        test_auc = (TP + TN) / (TP + TN + FP + FN)
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        test_auc = (TP + TN) / (TP + TN + FP + FN)
        F1_score = 2 * (precision * recall) / (precision + recall)

    return precision ,recall ,test_auc ,F1_score, TP, FP, TN, FN


def evaluate(data_iter, model, device):
    model.eval()

    model_predict_results = []
    y_truth = []

    with torch.no_grad():
        if config.load_graph:
            for x ,y, z in data_iter:
                x ,y, z = x.to( device ) ,y.to( device ), z.to( device )
                logits = model( x, z )
                # Original: directly argmax
                # model_predict_results.append(logits.argmax(1))
                # New: output all prob and calculate with threshold
                logits = F.softmax( logits ,dim=1 )
                model_predict_results.append( logits[: ,1] )
                y_truth.append( y )
        else:
            for x ,y in data_iter:
                x ,y = x.to( device ) ,y.to( device )
                z = 0
                logits = model(x, z)
                # Original: directly argmax
                # model_predict_results.append(logits.argmax(1))
                # New: output all prob and calculate with threshold
                logits = F.softmax( logits ,dim=1 )
                model_predict_results.append( logits[: ,1] )
                y_truth.append( y )

        model.train()
        model_predict_results = flatten( model_predict_results )
        y_truth = flatten( y_truth )
        return model_predict_results ,y_truth


if __name__ == '__main__':
    config = Config()
    train_model(config)
    """
    Epoch: 9, Batch: [410/469], Train loss 0.186, Train acc: 0.938
    Epoch: 9, Batch: [420/469], Train loss 0.150, Train acc: 0.938
    Epoch: 9, Batch: [430/469], Train loss 0.269, Train acc: 0.941
    Epoch: 9, Batch: [440/469], Train loss 0.197, Train acc: 0.925
    Epoch: 9, Batch: [450/469], Train loss 0.245, Train acc: 0.917
    Epoch: 9, Batch: [460/469], Train loss 0.272, Train acc: 0.902
    Accuracy on test 0.886
    """
