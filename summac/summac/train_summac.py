import logging
from typing import List

from torch.distributed.elastic.multiprocessing.errors import record
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from benchmark import SummaCBenchmark
import torch, nltk, numpy as np, argparse
from utils_optim import build_optimizer
from model_summac import SummaCConv, model_map
import os, time
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch import distributed as dist, Tensor
from nltk import data

data.path.append('/home/liwenbo/summac/summac/nltk_data/')
logger = get_logger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

@record
def main():
    parser = argparse.ArgumentParser()

    model_choices = list(model_map.keys()) + ["multi", "multi2"]

    parser.add_argument("--model", type=str, choices=model_choices, default="mnli")
    parser.add_argument("--granularity", type=str, default="sentence")
    # , choices=["sentence", "paragraph", "mixed", "2sents"]
    parser.add_argument("--pre_file", type=str, default="", help="If not empty, will use the precomputed instead of "
                                                                 "computing images on the fly. (useful for "
                                                                 "hyper-param tuning)")
    parser.add_argument("--bins", type=str, default="percentile", help="How should the bins of the histograms be "
                                                                       "decided (even%d or percentile)")
    parser.add_argument("--nli_labels", type=str, default="e", choices=["e", "c", "n", "ec", "en", "cn", "ecn"],
                        help="Which of the three labels should be used in the creation of the histogram")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of passes over the data.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Number of passes over the data.")
    parser.add_argument("--norm_histo", action="store_true", help="Normalize the histogram to be between 0 and 1, "
                                                                  "and include the explicit count")
    parser.add_argument("--size", type=int, default=5, help="num of data used to train")

    args = parser.parse_args()
    train(**args.__dict__)


def train(model="mnli", granularity="sentence", nli_labels="e", pre_file="", num_epochs=5, optimizer="adam",
          train_batch_size=32, learning_rate=0.1, bins="even50", silent=False, norm_histo=False, size=5):
    experiment = "%s_%s_%s_%s" % (model, granularity, bins, nli_labels)
    torch.manual_seed(0)
    np.random.seed(0)
    """init accelerator"""
    accelerator = Accelerator()
    logger.info(accelerator.state, main_process_only=True)
    if not silent:
        logger.info("Experiment name: %s" % (experiment), main_process_only=True)
    if len(pre_file) == 0:
        standard_pre_file = "./data/summac_cache/train_%s_%s.jsonl" % (model, granularity)
        if os.path.isfile(standard_pre_file):
            pre_file = standard_pre_file

    def sent_tok(text):
        sentences = nltk.tokenize.sent_tokenize(text)
        return [sent for sent in sentences if len(sent) > 10]

    def collate_func(inps):
        documents, claims, labels = [], [], []
        for inp in inps:
            if len(sent_tok(inp["claim"])) > 0 and len(sent_tok(inp["document"])) > 0:
                documents.append(inp["document"])
                claims.append(inp["claim"])
                labels.append(inp["label"])
        """convert to accelerator"""
        labels = torch.LongTensor(labels)  # labels = torch.LongTensor(labels).to(device)
        return documents, claims, labels

    def collate_pre(inps):
        documents = [inp["document"] for inp in inps]
        claims = [inp["claim"] for inp in inps]
        # images = [[np.array(im) for im in inp["image"]] for inp in inps]
        images = [np.array(inp["image"]) for inp in inps]
        """convert to accelerator"""
        labels = torch.LongTensor([inp["label"] for inp in inps])
        # labels = torch.LongTensor([inp["label"] for inp in inps]).to(device)
        return documents, claims, images, labels


    """load dataset"""
    # fastcc
    # save = False
    # fcb = SummaCBenchmark(cut="train", dataset_names=['factcc'], accelerator=accelerator)
    # d_train: List[dict] = fcb.get_dataset(dataset_name='factcc')
    # dl_train = DataLoader(dataset=d_train, batch_size=train_batch_size, pin_memory=True,
    #                       sampler=RandomSampler(d_train), num_workers=3,
    #                       collate_fn=collate_func)

    # cogensum
    save = False  # True
    fcb = SummaCBenchmark(cut="val", dataset_names=['cogensumm'])
    dataset: List[dict] = fcb.get_dataset(dataset_name='cogensumm')
    d_train = dataset[:int(len(dataset) * 0.9)]
    d_val = dataset[int(len(dataset) * 0.9):]
    # 最后10%是val set
    dl_train = DataLoader(dataset=d_train, shuffle=False,
                          batch_size=train_batch_size, collate_fn=collate_func)
    dl_val = collate_func(d_val)

    device = "cuda"  # "cpu" if precomputed else "cuda"
    if model == "multi":
        models = ["mnli", "anli", "vitc"]
    elif model == "multi2":
        models = ["mnli", "vitc", "vitc-only", "vitc-base"]
    else:
        models = [model]
    """build model"""
    model = SummaCConv(models=models, granularity=granularity, nli_labels=nli_labels,
                       device=device, bins=bins, norm_histo=norm_histo,
                       start_file="/home/liwenbo/summac/summac/summac/vitc_sentence_percentile_e_bacc0.700.bin",
                       save=save, acc=accelerator)
    optimizer = build_optimizer(model, learning_rate=learning_rate, optimizer_name=optimizer)
    if not silent:
        logger.info("Model Loaded", main_process_only=True)

    """
    convert to accelerate:
        model: model
        optimizer: optimizer
        data: dl_train
    """
    model, optimizer, dl_train = accelerator.prepare(model, optimizer, dl_train)
    model: SummaCConv
    optimizer: Optimizer
    logger.debug(f"device_info: model on {model.device}, dl_train on {dl_train.device}")
    if not silent:
        logger.info("Length of dataset. [Train: %d] [Valid: %d]" % (len(d_train), len(d_val)), main_process_only=True)
    criterion = torch.nn.CrossEntropyLoss()
    eval_interval = 8  # every _ batch in one epoch
    best_val_score = 0.0
    best_file = ""
    logger.info(f"start training on GPU={torch.cuda.get_device_name()}", main_process_only=True)
    # bar = tqdm(enumerate(range(num_epochs)), total=num_epochs, disable=not accelerator.is_local_main_process)
    for i, epoch in enumerate(range(num_epochs)):
        # bar.set_description(f"epoch: {epoch}/{num_epochs}")
        itr = enumerate(dl_train)
        if not silent:
            itr = tqdm(itr, total=len(dl_train), disable=not accelerator.is_local_main_process)
        for idx, batch in itr:
            itr.set_description(f"batch:{idx}/{len(dl_train)}")
            optimizer.zero_grad()
            documents, claims, batch_labels = batch
            logits, _, _ = model(idx=idx, originals=documents, generateds=claims)
            logits: Tensor
            loss = criterion(logits, batch_labels)
            if not silent:
                itr.set_postfix_str(f"epoch:{i}/{num_epochs}, loss={loss:.2f}")
            """convert to accelerator"""
            accelerator.backward(loss)  # loss.backward()
            optimizer.step()
            if (idx + 1) % eval_interval == 0:
                logger.info("start evaluating", main_process_only=True)
                eval_time = time.time()
                benchmark = fcb.evaluate(model)
                val_score = benchmark["overall_score"]
                eval_time = time.time() - eval_time
                if eval_time > 10.0:
                    model.module.save_imager_cache()
                if val_score > best_val_score:
                    best_val_score = val_score
                    # only main process handles files
                    if len(best_file) > 0 and accelerator.is_main_process:
                        os.remove(best_file)
                    best_file = "./%s_bacc%.3f.bin" % (experiment, best_val_score)
                    """convert to accelerator"""
                    # torch.save(model.state_dict(), best_file)
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(unwrapped_model.state_dict(), best_file)
                if not silent:
                    for t in benchmark["benchmark"]:
                        logger.info(
                            "[%s] Score: %.3f (thresh: %.3f)" % (t["name"].ljust(10), t["score"], t["threshold"]),
                            main_process_only=True)
    return best_val_score


if __name__ == "__main__":
    """
    accelerate launch --main_process_port 29508 --config_file /home/liwenbo/summac/summac_config.yaml train_summac.py --model vitc --granularity sentence --train_batch_size 16 --num_epochs 10 --nli_labels e 
    """
    main()
