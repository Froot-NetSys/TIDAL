import torch
import torch.nn.functional as F
from config import Config
from data_helper import LoadSentenceClassificationDataset
from ClassificationModel import ClassificationModel
import os
import numpy as np
# roc curve and auc
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
from matplotlib import pyplot
import random
import pandas as pd

seed = 100
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def flatten(xss):
    '''
    Flatten a list of sub lists
    e.g.:
    Input:  [[20180406113149], [20180406120452]]
    Output: [20180406113149, 20180406120452]
    '''
    return [x for xs in xss for x in xs]


def evaluate(data_iter, model, device):
    model.eval()

    model_predict_results = []
    y_truth = []

    with torch.no_grad():
        for x ,y, z in data_iter:
            x ,y, z = x.to( device ) ,y.to( device ), z.to( device )
            logits = model( x, z )
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


def cal_roc_curve(y_truth, y_model):
    #### ROC curve
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range( len( y_model ) )]
    # predict probabilities
    lr_probs = y_model

    testy = y_truth
    ns_auc = roc_auc_score( testy ,ns_probs )
    lr_auc = roc_auc_score( testy ,lr_probs )
    # summarize scores
    print( 'No Skill: ROC AUC=%.3f' % (ns_auc) )
    print( 'Logistic: ROC AUC=%.3f' % (lr_auc) )
    # calculate roc curves
    ns_fpr ,ns_tpr ,_ = roc_curve( testy ,ns_probs )
    lr_fpr ,lr_tpr ,_ = roc_curve( testy ,lr_probs )
    # plot the roc curve for the model
    pyplot.plot( ns_fpr ,ns_tpr ,linestyle='--' ,label='No Skill' )
    pyplot.plot( lr_fpr ,lr_tpr ,marker='.' ,label='Logistic' )
    # axis labels
    pyplot.xlabel( 'False Positive Rate' )
    pyplot.ylabel( 'True Positive Rate' )
    # show the legend
    pyplot.legend()
    pyplot.title("ROC curve")
    # show the plot
    # pyplot.show()

    return lr_fpr, lr_tpr


def cal_precision_recall_curve(y_truth, y_model):
    ##### Precision-Recall curve
    # predict probabilities
    lr_probs = y_model
    testy = y_truth
    testy = np.array( testy )

    p_index = [idx for idx ,val in enumerate( y_model ) if val > 0.5]
    y_test_by_model = np.zeros( len( y_model ) )
    y_test_by_model[p_index] = 1

    lr_precision ,lr_recall ,_ = precision_recall_curve( testy ,lr_probs )
    lr_f1 ,lr_auc = f1_score( testy ,y_test_by_model ) ,auc( lr_recall ,lr_precision )
    # summarize scores
    print( 'Logistic: f1=%.3f auc=%.3f' % (lr_f1 ,lr_auc) )
    # plot the precision-recall curves
    no_skill = len( testy[testy == 1] ) / len( testy )
    pyplot.plot( [0 ,1] ,[no_skill ,no_skill] ,linestyle='--' ,label='No Skill' )
    pyplot.plot( lr_recall ,lr_precision ,marker='.' ,label='Logistic' )
    # axis labels
    pyplot.xlabel( 'Recall' )
    pyplot.ylabel( 'Precision' )
    # show the legend
    pyplot.legend()
    pyplot.title( "Precision-Recall curve" )
    # show the plot
    # pyplot.show()

    return lr_recall ,lr_precision


if __name__ == '__main__':
    config = Config()
    model_save_path = 'saved_model/full_model/trace_full_model.pt'
    classification_model =  ClassificationModel(vocab_size=config.vocab_size,
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
            loaded_paras = torch.load(model_save_path)
        else:
            loaded_paras = torch.load( model_save_path, map_location=torch.device('cpu'))
        classification_model.load_state_dict(loaded_paras)
        print("Load the model...")
        # print(classification_model)

    data_loader = LoadSentenceClassificationDataset( config.train_corpus_file_paths ,
                                                     batch_size=config.batch_size ,
                                                     min_freq=config.min_freq ,
                                                     max_sen_len=config.max_sen_len )
    _ ,test_iter = data_loader.load_train_val_test_data(
        config.train_corpus_file_paths ,config.test_corpus_file_paths )

    y_model, y_truth = evaluate( test_iter ,classification_model ,config.device )

    lr_fpr, lr_tpr = cal_roc_curve(y_truth, y_model)
    lr_recall ,lr_precision = cal_precision_recall_curve(y_truth, y_model)

    df_roc = pd.DataFrame( {"lr_fpr": lr_fpr ,"lr_tpr": lr_tpr} )
    df_pr = pd.DataFrame( {"lr_recall": lr_recall ,"lr_precision": lr_precision} )
    df = pd.concat( [df_roc ,df_pr] ,axis=1 )

    df.to_csv( 'our_result_trace.csv' ,index=False )