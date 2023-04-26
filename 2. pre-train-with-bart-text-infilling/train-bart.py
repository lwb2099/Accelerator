from types import SimpleNamespace
from transformers import BartForConditionalGeneration, AdamW, get_scheduler
from data_loader import Loading_Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from accelerate import Accelerator
import argparse
"""
Train Bart in text-infilling task, we will use book-corpuse dataset
"""

# * parameters for training
config = SimpleNamespace(    
        num_training_steps = 100,
        warm_up_steps = 10,
        model_name = "bart-base",
        val_interval = 10,
        seed = 42,
        lr = 1e-3,
        batch_size = 5,
        dataset_name = "bookcorpus",
        val_step = 10,
        model_ckpt_path='./models/',
        train_size = 10**5,
        val_size=10**3,
        test_size=10**3,
    )



class Bart:
    def __init__(self) -> None:
        pass
    

    def prepare(self, args):
        #* set seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        #* load data
        train_dataloader, val_dataloader, test_dataloader = Loading_Dataset(args)
        #* load_pre-trained model
        model = BartForConditionalGeneration.from_pretrained("bart-base")
        #* set up optimizer
        optimizer = AdamW(model.parameters, lr = args.lr)
        lr_scheduler = get_scheduler(
                                    "linear",
                                    optimizer=optimizer,
                                    num_warmup_steps=args.warm_up_steps,
                                    num_training_steps=args.num_training_steps
                                )
        self.accelerator = Accelerator()
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.test_dataloader, self.lr_scheduler \
        = self.accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, test_dataloader, lr_scheduler)

    
    def fit(self, args):
        progress_bar = tqdm(range(1, args.num_training_steps+1))
        for epoch in progress_bar:
            progress_bar.set_description(f"Epoch: {epoch}/num_training_steps")
            self._one_epoch(train=True)
        
        if args.val_interval != 0 and epoch % args.val_interval == 0:
            self._one_epoch(train=False)

        self._save_model(args.model_ckpt_path)

    def _one_epoch(self, train=True):
        if train: self.model.train()
        else: self.model.eval()
        progress_bar = tqdm(self.train_dataloader) if train else tqdm(self.val_dataloader)
        for i, (masked_text, text) in enumerate(progress_bar):
            pred_text = self.model(masked_text)
        # ... to be finished
    
    def _save_model(self, model_ckpt_path):
        torch.save(self.model.state_dict(), os.path.join("models", "model_ckpt.pt"))


def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)
  
if __name__ == "__main__":
    parse_args(config)

    # * seed everything

    diffuser = Bart()
    diffuser.prepare(config)
    diffuser.fit(config)

    
    