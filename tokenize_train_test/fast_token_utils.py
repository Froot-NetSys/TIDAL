import time
import pandas as pd
from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict

global layer_dic
layer_dic = {}
maindic = {}

TREE_PATH = '../data/trace/tokenized_train_test/trace_tree_no_attack.txt'
# TREE_PATH = '../data/param_tradeoff/tree_diff_threshold/trace_tree_100.txt'

def attach(maindic ,layer ,ancestors, tree_path = TREE_PATH):
    global layer_dic, input_
    '''
    Insert a branch of directories on its trunk.
    # '''
    FILE_MARKER = '<files>'
    # TODO: read tree file as a arg plz!!!
    with open( tree_path ,'r' ) as file:
        input_ = file.read()

    if layer not in layer_dic:
        layer_dic[layer] = 1
    else:
        layer_dic[layer] += 1

    dic = {}
    for line in input_.split( '\n' ):
        branch = line
        parts = branch.split( '/' )
        a = ''
        num_of_line = 1
        if len( parts ) > layer:
            for i in range( layer ):
                a += parts[i] + '/'
            if a == ancestors:
                if parts[layer] not in dic:
                    dic[parts[layer]] = num_of_line
                else:
                    dic[parts[layer]] += num_of_line

    for key in dic:
        a = ancestors + key + "/"
        maindic[key] = dict()
        attach( maindic[key] ,layer + 1 ,a, tree_path=TREE_PATH)

    return maindic


maindic = attach( maindic ,1 ,ancestors='/', tree_path=TREE_PATH)


def find_token_root_path(each_seq):
    # import pdb; pdb.set_trace()
    # if it is srcUUID or dstUUID, just return the original value
    if each_seq.isdigit():
        return int(each_seq)
    # if itself is already in token list, just use it
    if each_seq in input_:
        return each_seq

    split_seq = each_seq.split( '/' )

    dic = maindic
    if split_seq[0] == '':
        acc = ''
    elif split_seq[0].startswith( 'EVENT_' ):
        return each_seq
    elif split_seq[0] in maindic:
        acc = split_seq[0]
    else:
        # if each_seq is just one single file, eg: kernel_mem.txt
        return 'other_file'

    def find_acc(split_seq ,dic ,acc):
        if len( split_seq ) == 0:
            return acc
        elif split_seq[0] not in dic:
            return acc + '/'
        else:
            if len( split_seq ) == 1:
                return acc + '/'
            path = find_acc( split_seq[1:] ,dic[split_seq[0]] ,split_seq[0] )
            return acc + '/' + path

    return find_acc( split_seq[1:] ,dic ,acc )


def token_each_seq(seq):
    def find_matched_token_number(each_root):
        # last check: if each_root is NOT in df['path'], mark it as "other_file",
        # because it appears less than threshold
        if isinstance(each_root, int):
            # if it is srcUUID or dstUUID, just return the original value
            token_num = each_root
        elif each_root in df['path'].to_list():
            token_num = df[df['path'] == each_root].tokenized_value.to_list()[0]
        else:
            token_num = df[df['path'] == 'other_file'].tokenized_value.to_list()[0]

        return token_num

    # TODO: change the tree.txt to an input args
    with open(TREE_PATH ,'r' ) as file:
        all_file_keywords = file.read().splitlines()

    df = pd.DataFrame()
    df['path'] = all_file_keywords
    df['tokenized_value'] = df.index.to_list()

    seq_list = seq.split( '\t' )
    seq_df = pd.DataFrame()
    seq_df['original_seq'] = seq_list
    seq_df['token_root'] = seq_df['original_seq'].apply( lambda x: find_token_root_path( x ) )
    seq_df['token_num'] = seq_df['token_root'].apply( lambda x: find_matched_token_number( x ) )

    return seq_df['token_num'].to_list()


def build_hashmap_with_csv(audit_log_file_path):
    '''
    Prepare a timestamp_hash and subject_statements for later use.

    :param audit_log_file_path: input data path
    :return:
        raw_dataframe: the dataframe of all input data
        subject_statements: all lines of original log, without 'timestamp' and 'uuid'
        subjects: all unique subjects (or events)
        timestamp_hash: the hashmap dataframe where key is 'timestamp', value is raw data line numbers of
                        current timestamp. e.g.: {'20180406112100': [100, 101, ..., 167]}
    '''
    global subject_statements

    # formatting
    df = pd.read_csv(audit_log_file_path)
    raw_dataframe = df

    # final return data
    subject_statements = df.drop( 'timestamp' ,1 )
    subject_statements = subject_statements.drop( 'truth_label' ,1 )
    # Only when we have graph data
    # import pdb; pdb.set_trace()
    # if load_graph == False:
    subject_statements = subject_statements.drop( 'data' ,1 )

    subjects = raw_dataframe['target'].unique()

    # Build the hashmap based on timestamp
    timestamp_hash = df.groupby( 'timestamp' ).groups
    # convert the values to pandas
    timestamp_hash = {k: v.tolist() for k ,v in timestamp_hash.items()}
    timestamp_hash = pd.DataFrame( list( timestamp_hash.items() ) ,columns=['timestamp' ,'indices'] )

    return raw_dataframe ,subject_statements ,subjects ,timestamp_hash


def sequenize_indices(indices):
    '''
    Goal: for each line in the timestamp_hashmap, map the indices back to the original logs (from subject_statements)
    e.g.: {'20180406112100': [100, 101, ..., 167]} ----->
          {'20180406112100': ['uuid	EVENT_OPEN other_file	uuid EVENT_OPEN	lib_file ...']}

    :param indices: line indices of each timestamp
    :return: original logs of each timestamp
    '''
    global subject_statements

    result = subject_statements \
        .loc[indices] \
        .to_csv(header=None, index=False, sep='\t') \
        .replace('\n', '\t').strip()
    return result


def main():
    input = 'active/68A1E29D16'
    asw = find_token_root_path(input)
    print(asw, "---ouput test")


if __name__ == '__main__':
    main()