from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BartTokenizer

def Loading_Dataset(args):
    dataset = load_dataset(args.dataset_name)
    tokenizer =BartTokenizer.from_pretrained(args.model_name)
    def preprocess(dataset):
        """random mask sequence"""
        pass


    # * set up dataset
    dataset = dataset.map(preprocess, batched=True)
    sub_train_set = dataset['train'].shuffle(args.seed).select(range(args.train_size))
    sub_val_set = dataset['validation'].shuffle(args.seed).select(range(args.val_size))
    sub_test_set = dataset['test'].shuffle(args.seed).select(range(args.test_size))
    train_dataloader = DataLoader(sub_train_set, shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(sub_val_set, batch_size=args.batch_size)
    test_dataloader = DataLoader(sub_test_set, batch_size=args.batch_size)
    return train_dataloader, valid_dataloader, test_dataloader
