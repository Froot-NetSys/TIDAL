import numpy as np
import fast_token_utils as utils
from tqdm import tqdm
import pandas as pd
import time
import random
import os
import json
import argparse
import csv
from itertools import groupby
from ast import literal_eval

import warnings
warnings.filterwarnings("ignore")

# TODO: after chunk the UUID does not always match
SEQ_CHUNK_LENGTH = 1024
LOAD_GRAPH = True

seed = 100
np.random.seed(seed)
random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Generating train.json and test.json script.")
    parser.add_argument( "--semi_data_path" ,type=str ,required=True ,
                         help='The data output folder for each semi-SL run.' )
    return parser.parse_args()

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)

def remove_recurring_token(originList):
    "[1, 2, 1, 2, 5, 4] --> [1, 2, 5, 4]"
    x = grouped( originList ,2 )
    new_list = []
    for i in x:
        new_list.append( i )

    remove_recurring_list = [key for key ,_group in groupby( new_list )]
    remove_recurring_list = list( sum( remove_recurring_list ,() ) )

    return remove_recurring_list

def seq_chunk(seq):
    size = SEQ_CHUNK_LENGTH
    result = []
    chunked_seq = (seq[pos:pos + size] for pos in range( 0 ,len( seq ) ,size ))

    for x in chunked_seq:
        result.append( x )

    return result

def build_both_samples(raw_df ,timestamp_hash, type_flag=None):
    '''
    Build non-attack (label=0) training samples.

    :param non_attack_timestamp: list of timestamps that are OUT of the attack period.
    :param timestamp_hash: the hashmap dataframe where key is 'timestamp', value is raw data line numbers of
                            current timestamp. e.g.: {'20180406112100': [100, 101, ..., 167]}
    :return: tokenized non-attack dataframe
    '''
    raw_df = raw_df.drop_duplicates( subset='timestamp' ,keep="first" )
    raw_df = raw_df.drop( 'action' ,1 )
    raw_df = raw_df.drop( 'target' ,1 )

    if type_flag == 'ATTACK':
        print("==================Extract attack samples==================")
        extract_df = raw_df[raw_df['truth_label'] == 1]
    elif type_flag == 'BENIGN':
        print("==================Extract benign samples==================")
        extract_df = raw_df[raw_df['truth_label'] == 0]
    else:
        raise ValueError('You can only choose ATTACK or BENIGN!!!')

    extract_df['timestamp'] = extract_df['timestamp'].apply( lambda x: tuple( [x] ) )

    tqdm.pandas( desc='Merge indices for each combo' )
    extract_df["selected_indices"] = extract_df['timestamp'].progress_apply( lambda x:
                                                                           sum( timestamp_hash.loc[
                                                                                    timestamp_hash['timestamp'].isin(
                                                                                        x )].indices.tolist() ,[] ) )

    extract_df['selected_indices'] = extract_df['selected_indices'].apply( lambda x: sorted( x ) )

    tqdm.pandas( desc='Find original log with indices' )
    extract_df['seq'] = extract_df['selected_indices'].progress_apply( lambda x: utils.sequenize_indices( x ) )

    tqdm.pandas( desc='Tokenize log into num' )
    extract_df['tokenized_num'] = extract_df['seq'].progress_apply( lambda x: utils.token_each_seq( x ) )

    tqdm.pandas( desc='Chunk each timestamp to fix-len, explode it' )
    extract_df['tokenized_num_chunk'] = extract_df['tokenized_num'].progress_apply( lambda x: seq_chunk( x ) )

    extract_df = extract_df.drop( ['timestamp' ,'selected_indices' ,'seq' ,'tokenized_num' ,] ,axis=1 )

    extract_df = extract_df.apply( pd.Series.explode ).reset_index()

    extract_df['tokenized_num_chunk'] = extract_df['tokenized_num_chunk'].apply( lambda x: tuple( x ) )

    if LOAD_GRAPH:
        extract_df['data'] = extract_df['data'].progress_apply( lambda x: literal_eval( x ) )

    extract_df = extract_df.drop_duplicates( subset='tokenized_num_chunk' ,keep="first" )

    # Length check:
    tmp = extract_df['tokenized_num_chunk'].str.len()
    print( "Find non-attack samples: " ,len( tmp ) )
    print( "Max length: " ,tmp.max() ,"\nMin length: " ,tmp.min() ,"\nAvg length: " ,tmp.sum() / len( tmp ) )

    return extract_df

def resample(semi_data_path):
    '''
    Re-sample the attack (label=1) data because of imbalance.
    Logic:
        if label=0 data is about n times of label=1 data,
        repeat all label=1 data n times.
        The after resample portion would be close to 1:1, not exactly 1:1

    :param semi_data_path: the sub folder of semi-SL experiment. e.g.: 'true_label_0.2_ratio'
    :return: output re-sampled data into 'semi_data_path/resampling/resampling.json'
    '''

    global repeat_amount

    non_sampling_path = semi_data_path + 'semi/true_train.json'
    with open( non_sampling_path ,'r' ) as f:
        x_train ,y_train, z_train = json.load( f )

    if LOAD_GRAPH:
        df = pd.DataFrame( {'x_train': x_train ,'y_train': y_train, 'z_train': z_train} )
    else:
        z_train = y_train
        df = pd.DataFrame( {'x_train': x_train ,'y_train': y_train, 'z_train': z_train} )

    df['y_train'].value_counts()
    count_y_1 = df['y_train'].value_counts().get( 1 )
    count_y_0 = df['y_train'].value_counts().get( 0 )
    print( "number of 0 label:" + str( count_y_0 ) )
    print( "number of 1 label:" + str( count_y_1 ) )

    if count_y_1 < count_y_0:
        repeat_amount = int( count_y_0 / count_y_1 )
    else:
        repeat_amount = 1

    df_attack = df[df['y_train'] == 1]
    df_non_attack = df[df['y_train'] == 0]

    # pure repeat
    df_attack = df_attack.loc[df_attack.index.repeat( repeat_amount )]

    df = pd.concat( [df_attack ,df_non_attack] ,axis=0 )
    print( "after resampling:" )
    count_y_1 = df['y_train'].value_counts().get( 1 )
    count_y_0 = df['y_train'].value_counts().get( 0 )
    print( "number of 0 label:" + str( count_y_0 ) )
    print( "number of 1 label:" + str( count_y_1 ) )

    x_train_resample = df['x_train'].to_list()
    y_train_resample = df['y_train'].to_list()
    z_train_resample = df['z_train'].to_list()
    x_y_z_list = [x_train_resample ,y_train_resample, z_train_resample]

    resampling_out = open( semi_data_path + 'semi/train.json' ,'w' )
    json.dump( x_y_z_list ,resampling_out )
    resampling_out.close()

def main():
    args = parse_args()

    semi_data_path = args.semi_data_path

    # Init the training and testing data for multiple data sets
    x_train_all = []
    y_train_all = []
    z_train_all = []

    x_test_all = []
    y_test_truth_all = []
    z_test_all = []

    # makedir for data resampling
    os.makedirs( semi_data_path + 'semi/' ,exist_ok=True )

    # Prepare the true-label data
    for file in os.listdir( semi_data_path ):
        if file.startswith( "true_label" ):
            print( "-------Find true label file: " + file )

            input_file_path = semi_data_path + file

            raw_dataframe ,subject_statements ,subjects ,timestamp_hash = utils.build_hashmap_with_csv(
                input_file_path)
            attack_df = build_both_samples( raw_dataframe ,timestamp_hash, type_flag='ATTACK' )
            non_attack_df = build_both_samples( raw_dataframe ,timestamp_hash, type_flag='BENIGN' )

            merge_df = pd.concat( [attack_df ,non_attack_df] )
            # Drop mislabeled seq:
            # If there are seq marked as both 0 and 1, then mark them all by 0
            merge_df[merge_df.duplicated( subset='tokenized_num_chunk' ,keep=False )] = merge_df[
                merge_df.duplicated( subset='tokenized_num_chunk' ,keep=False )].replace( 0 ,1 )

            x_train_all.extend( merge_df['tokenized_num_chunk'].to_list() )
            y_train_all.extend( merge_df['truth_label'].to_list() )
            if LOAD_GRAPH:
                z_train_all.extend( merge_df['data'].to_list() )

        if file.startswith( "pseudo_label" ):
            print( "-------Find pseudo label file: " + file )

            input_file_path = semi_data_path + file

            raw_dataframe ,subject_statements ,subjects ,timestamp_hash = utils.build_hashmap_with_csv(
                input_file_path)
            attack_df = build_both_samples( raw_dataframe ,timestamp_hash ,type_flag='ATTACK' )
            non_attack_df = build_both_samples( raw_dataframe ,timestamp_hash ,type_flag='BENIGN' )

            merge_df = pd.concat( [attack_df ,non_attack_df] )
            # Drop mislabeled seq:
            # If there are seq marked as both 0 and 1, then mark them all by 1
            merge_df[merge_df.duplicated( subset='tokenized_num_chunk' ,keep=False )] = merge_df[
                merge_df.duplicated( subset='tokenized_num_chunk' ,keep=False )].replace( 0 ,1 )

            x_test_all.extend( merge_df['tokenized_num_chunk'].to_list() )
            y_test_truth_all.extend( merge_df['truth_label'].to_list() )
            if LOAD_GRAPH:
                z_test_all.extend( merge_df['data'].to_list() )

    # Write out the prepared testing data
    test_x_y_z_list = [x_test_all ,y_test_truth_all ,z_test_all]
    nonsampling_out = open( semi_data_path + "semi/pseudo_train.json" ,'w' )
    json.dump( test_x_y_z_list ,nonsampling_out )
    nonsampling_out.close()
    print( "Saved pseudo_train.json file ..." )

    # Write out the true-label prepared training data
    x_y_z_list = [x_train_all ,y_train_all, z_train_all]
    nonsampling_out = open( semi_data_path + "semi/true_train.json" ,'w' )
    json.dump( x_y_z_list ,nonsampling_out )
    nonsampling_out.close()
    print( "Saved true_train.json file ..." )

    # Resample training data because it's highly imbalanced
    resample( semi_data_path )


if __name__ == '__main__':
    main()