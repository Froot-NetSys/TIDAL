# Train the multi-head transformer

## Setup
```
torch > 1.10.0 
```

## Update the config for different dataset
TODO: add a flexible parser
```
self.dataset_dir = os.path.join(self.project_dir, '../data/cadet/train_data')
self.model_name = 'sketch_model.pt'
self.vocab_size = 4390
```

## Train the model
```
python train.py --semi_data_path="data/trace/" --log_path='log_trace.csv' --token_size=162

python train.py --semi_data_path="data/fived/" --log_path='log_fived.csv' --token_size=145

python train.py --semi_data_path="data/clearscope/" --log_path='log_clearscope.csv' --token_size=101

python train.py --semi_data_path="data/theia/" --log_path='log_theia.csv' --token_size=140

python train.py --semi_data_path="data/cadet/" --log_path='log_cadet.csv' --token_size=363
```

## Test the model
Update the "model_save_path"
```
python test_log.py 
```

## Test the varied length seq separately
Split the original test data into "short, mid, long"
```
python split_test_with_len.py --semi_data_path='data/cadet/'
```

## Test the model with ROC and PRC curve
Update the "model_save_path"
```
python test_roc.py 
```