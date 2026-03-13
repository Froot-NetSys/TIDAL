import pandas as pd
import os

# sketch.txt number+1
SKETCH_NUM = 1602
INTERVAL = 1
DATASET = '../data/param_tradeoff/'
OUTPUT_LOG = 'full_data/graph/processed_3_hop_trace.csv'

# Read processed cadets data
data = pd.read_csv(DATASET + 'full_data/no_graph/processed_uuid_trace.txt', sep="\t", header=None)
data.columns = ["srcUUID", "dstUUID", "action", "target", "timestamp"]

df = data.drop('srcUUID', axis=1)
df = df.drop('dstUUID', axis=1)

df['flag'] = df.timestamp >= INTERVAL + df.timestamp.shift()

cols = ["action", "target", "timestamp", 'flag']
df = df[cols]

df['grp'] = df['flag'].cumsum()
df = df.drop('flag', axis=1)

# Read sketch data into a DF {index: grp_id, data: sketch}
all_sketch = []
for i in range(0, SKETCH_NUM):
    print(i, "------count")
    file_name = DATASET + '3_hop_train_toy/sketch-toy-' + str(i) + '.txt'
    if os.path.isfile( file_name ):
        file = open(file_name, 'r')
        lines = file.readlines()
        if len(lines) > 1:
            tmp_data = lines[1].split()
        else:
            tmp_data = []
        each_sketch_data = list( map( int ,tmp_data ) )
        all_sketch.append( each_sketch_data )

# import pdb; pdb.set_trace()

# trace: df.loc[(df['timestamp'] >= 20180412133600) & (df['timestamp'] <= 20180412134000)] --> 309--362

sketch_df = pd.DataFrame()
sketch_df['data'] = all_sketch
sketch_df['grp'] = sketch_df.index

merged_df = df.merge(sketch_df, how='left')
merged_df = merged_df.drop('grp', axis=1)

# drop the lines without graph data
merged_df = merged_df[merged_df['data'].map(lambda d: len(d)) > 0]

merged_df.to_csv(DATASET + OUTPUT_LOG, index=False)

