import math
from datasets import load_dataset
from transformers import AutoTokenizer, BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
import numpy as np
import evaluate
from transformers import DataCollatorForLanguageModeling
import os

"""
Train bart on text-infilling task with huggingface wrapped pipelines
example code when writing this script:
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb#scrollTo=YbSwEhQ63l_L
"""

os.environ['CUDA_VISIBLE'] = "0,1,2"
MAX_LENGTH = 256
train_size = 10**3
val_size =  10**1
test_size = val_size
if __name__ == '__main__':
    """1. data preprocess"""
    dataset = load_dataset("bookcorpus", cache_dir="./datasets")
    dataset = dataset.shuffle(seeds=42)
    model_name = "facebook/bart-large"
    tokenizer = BartTokenizer.from_pretrained(model_name, cached_dir="./models")

    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    # tokenized_datasets = dataset.map(tokenize, batched=True)
    print("tokenizing training dataset...")
    # small_train_dataset = tokenize(dataset["train"].select(range(10**5)))
    small_train_dataset = (dataset["train"].select(range(train_size))).map(tokenize, batched=True, num_proc=12, remove_columns=['text'])
    print("tokenizing val dataset...")
    small_eval_dataset = (dataset["train"].select(range(train_size, train_size+val_size))).map(tokenize, batched=True, num_proc=12, remove_columns=['text'])
    print("tokenizing test dataset...")
    small_test_dataset = (dataset["train"].select(range(train_size+val_size, train_size+val_size+test_size))).map(tokenize, batched=True, num_proc=12, remove_columns=['text'])
    
    # * automatically mask tokens
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.3)
    model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir="./models")

    """2. params defination"""
    training_args = TrainingArguments(
                                        per_device_train_batch_size=4,
                                        per_device_eval_batch_size=4,
                                        num_train_epochs=10,
                                        warmup_steps=100,
                                        save_steps=100,
                                        output_dir="./models/",
                                        overwrite_output_dir=True,
                                        evaluation_strategy = "epoch",
                                        learning_rate=2e-5,
                                        weight_decay=0.01,
                                        fp16=True,
                                        dataloader_num_workers=3,
                                        resume_from_checkpoint=True,
                                    )
    """3. build trainer"""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        data_collator=data_collator,
    )
    """4. train"""
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    predictions = trainer.predict(test_dataset=small_test_dataset)
    print(repr(predictions))

