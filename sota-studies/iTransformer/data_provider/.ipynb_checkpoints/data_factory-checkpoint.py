from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred
from torch.utils.data import DataLoader

#functions for process dataloader
def get_paired_data(root_path):
        input_path = os.path.join(root_path, "process_inputs.npy")
        target_path = os.path.join(root_path, "quality_targets.npy") 

        f = open(input_path, "rb")
        inputs = np.load(f)

        f = open(target_path, "rb")
        targets = np.load(f)

        paired_data = [[inputs[i], targets[i]] for i in range(len(targets))]
        print("Input-target pairs are loaded.")
        return paired_data
    
def get_dataloader(paired_data):

    # create pytorch datalaoders
    # Convert the paired data to tensors
    train_inputs = torch.tensor([pair[0] for pair in train_paired_data]).float()
    train_targets = torch.tensor([pair[1] for pair in train_paired_data]).float()

    # Create TensorDataset
    train_dataset = TensorDataset(train_inputs, train_targets)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq
    
    if args.data != "process":
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        
    else: # process prediction
        paired_data = get_paired_data(args.root_path)
        data_set = None
        data_loader = get_dataloader(paired_data)
    return data_set, data_loader
