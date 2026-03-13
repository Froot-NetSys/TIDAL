import torch
import torch.nn.functional as F
from config import Config
from data_helper import LoadSentenceClassificationDataset
from ClassificationModel import ClassificationModel
import os
import numpy as np
import pandas as pd
from itertools import zip_longest
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

seed = 100
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

def grouper_and_replace(iterable, n, fillvalue=None):
    '''
    Group the original results for different granularity.
    If there is one attack, then this group is marked as attack.
    e.g.:
    [[0, 0, 0], [0, 1, 0]] --> [0, 1]
    Args:
        iterable: the original list
        n: int value, chunk size
        fillvalue: if need to append a fill value.

    Returns: the grouped and updated results.
    '''
    replaced_list = []
    args = [iter(iterable)] * n
    group_list = list(zip_longest(*args, fillvalue=fillvalue))

    for i in group_list:
        if 1 in i:
            replaced_list.append(1)
        else:
            replaced_list.append(0)
    return replaced_list

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
            print(i, "-----index")
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
        precision = round(TP / (TP + FP), 2)
        recall = round(TP / (TP + FN), 2)
        test_auc = round((TP + TN) / (TP + TN + FP + FN), 2)
        F1_score = round(2 * (precision * recall) / (precision + recall), 2)
    return precision ,recall ,test_auc ,F1_score, TP, FP, TN, FN


def log_result(output_path, y_truth, group_value=None):
    precision_lst ,recall_lst ,test_auc_lst ,F1_score_lst ,TP_lst ,FP_lst ,TN_lst ,FN_lst = [] ,[] ,[] ,[] ,[] ,[] ,[] ,[]

    # start testing with different threshold
    SELF_TRAINING_Threshold = [0.1 ,0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9]
    # SELF_TRAINING_Threshold = [0.5]
    for x in SELF_TRAINING_Threshold:
        p_index = [idx for idx ,val in enumerate( y_model ) if val > x]
        y_test_by_model = np.zeros( len( y_model ) )
        y_test_by_model[p_index] = 1

        if group_value:
            y_truth_group = grouper_and_replace( y_truth ,group_value )
            y_test_by_model_group = grouper_and_replace( y_test_by_model ,group_value )
            precision ,recall ,test_auc ,F1_score ,TP ,FP ,TN ,FN = perf_measure( y_truth_group ,
                                                                                  y_test_by_model_group)
        else:
            precision ,recall ,test_auc ,F1_score ,TP ,FP ,TN ,FN = perf_measure( y_truth ,
                                                                                  y_test_by_model )
        precision_lst.append( precision )
        recall_lst.append( recall )
        test_auc_lst.append( test_auc )
        F1_score_lst.append( F1_score )
        TP_lst.append( TP )
        FP_lst.append( FP )
        TN_lst.append( TN )
        FN_lst.append( FN )

    log_df = pd.DataFrame( {
        'threshold': SELF_TRAINING_Threshold,
        'precision': precision_lst ,
        'recall': recall_lst,
        'auc': test_auc_lst ,
        'F1': F1_score_lst ,
        'TP': TP_lst ,
        'FP': FP_lst ,
        'TN': TN_lst ,
        'FN': FN_lst} )

    log_df.to_csv( output_path, index=False )
    return log_df

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
    model_save_path = 'saved_model/full_model/theia_full_model.pt'
    classification_model =  ClassificationModel(load_graph=config.load_graph,
                                                vocab_size=config.vocab_size,
                                                node_vec_size=config.node2vec_size,
                                                d_model=config.d_model,
                                                nhead=config.num_head,
                                                num_encoder_layers=config.num_encoder_layers,
                                                dim_feedforward=config.dim_feedforward,
                                                dim_classification=config.dim_classification,
                                                num_classification=config.num_class,
                                                dropout=0)
    # How to load model here??
    if os.path.exists(model_save_path):
        if torch.cuda.is_available():
            loaded_paras = torch.load(model_save_path, map_location=torch.device('cpu'))
            classification_model.load_state_dict( loaded_paras )
            classification_model.to( 'cuda' )

        else:
            loaded_paras = torch.load( model_save_path, map_location=torch.device('cpu'))
            classification_model.load_state_dict( loaded_paras )

        print("Load the model...")

    data_loader = LoadSentenceClassificationDataset( batch_size=config.batch_size ,
                                                     min_freq=config.min_freq ,
                                                     max_sen_len=config.max_sen_len )
    _ ,test_iter = data_loader.load_train_val_test_data(
        config.train_corpus_file_paths ,config.test_corpus_file_paths )

    y_model, y_truth = evaluate( test_iter ,classification_model ,config.device )

    _ = log_result('results/original_seq_test.csv', y_truth)
    # _ = log_result( 'results/group_10_test.csv' ,y_truth, group_value=10 )
    # _ = log_result( 'results/group_20_test.csv' ,y_truth, group_value=20 )
