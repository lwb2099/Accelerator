import logging
from typing import List

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

# dist.init_process_group("gloo")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
logger = get_logger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.DEBUG,
)

def train(model="mnli", granularity="sentence", nli_labels="e", pre_file="", num_epochs=5, optimizer="adam",
          train_batch_size=32, learning_rate=0.1, bins="even50", silent=False, norm_histo=False, size=5):
    experiment = "%s_%s_%s_%s" % (model, granularity, bins, nli_labels)
    """init accelerator"""
    accelerator = Accelerator(split_batches=True)
    logger.info(accelerator.state, main_process_only=False)
    if not silent:
        logger.debug("Experiment name: %s" % (experiment), main_process_only=True)
    if len(pre_file) == 0:
        standard_pre_file = "./data/summac_cache/train_%s_%s.jsonl" % (model, granularity)
        if os.path.isfile(standard_pre_file):
            pre_file = standard_pre_file

    device = "cuda"  # "cpu" if precomputed else "cuda"
    if model == "multi":
        models = ["mnli", "anli", "vitc"]
    elif model == "multi2":
        models = ["mnli", "vitc", "vitc-only", "vitc-base"]
    else:
        models = [model]
    """build model"""
    model = SummaCConv(models=models, granularity=granularity, nli_labels=nli_labels,
                       device=device, bins=bins, norm_histo=norm_histo)
    optimizer = build_optimizer(model, learning_rate=learning_rate, optimizer_name=optimizer)
    if not silent:
        logger.debug("Model Loaded", main_process_only=True)

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
    fcb = SummaCBenchmark(cut="val", dataset_names=['cogensumm'])
    d_train: List[dict] = fcb.get_dataset(dataset_name='cogensumm')

    dl_train = DataLoader(dataset=d_train[:max(len(d_train), size)], shuffle=False,
                          batch_size=train_batch_size, collate_fn=collate_func)
    #                     sampler=RandomSampler(d_train),
    # dl_train = load_data(split="train", rows=100, train_batch_size=train_batch_size)
    """
    convert to accelerate:
        model: model
        optimizer: optimizer
        data: dl_train
    """
    model, optimizer, dl_train = accelerator.prepare(model, optimizer, dl_train)
    model: SummaCConv
    optimizer: Optimizer
    dl_train: Tensor
    logger.debug(f"device_info: model on {model.device}, dl_train on {dl_train.device}")
    if not silent:
        logger.debug("Length of dataset. [Training: %d]" % (len(dl_train)), main_process_only=True)
    criterion = torch.nn.CrossEntropyLoss()
    eval_interval = 200
    best_val_score = 0.0
    best_file = ""
    logger.info(f"start training on GPU={torch.cuda.get_device_name()}", main_process_only=True)
    bar = tqdm(range(num_epochs), disable=not accelerator.is_local_main_process)
    for i, epoch in enumerate(bar):
        bar.set_description(f"epoch: {epoch}/{num_epochs}")
        itr = enumerate(dl_train)
        if not silent:
            itr = tqdm(itr, total=len(dl_train), disable=not accelerator.is_local_main_process)
        for idx, batch in itr:
            if not silent:
                itr.set_description(f"batch: {idx}/{len(dl_train)}")
            optimizer.zero_grad()
            documents, claims, batch_labels = batch
            logits, _, _ = model(idx=idx, originals=documents, generateds=claims)
            logits: Tensor
            logger.debug(f"logits: {logits.shape}, batch_labels: {batch_labels.shape}")
            loss = criterion(logits, batch_labels)

            """convert to accelerator"""
            accelerator.backward(loss)  # loss.backward()
            optimizer.step()
            if (idx + 1) % eval_interval == 0:
                logger.debug("start evaluating", main_process_only=True)
                eval_time = time.time()
                score = fcb.evaluate(model)
                benchmark = score["scores"]
                val_score = benchmark["overall_score"]
                eval_time = time.time() - eval_time
                if eval_time > 10.0:
                    model.save_imager_cache()
                if not silent:
                    itr.set_description("[Benchmark Score: %.3f]" % val_score)
                if val_score > best_val_score:
                    best_val_score = val_score
                    if len(best_file) > 0:
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
    """
    python train_summac.py --model vitc --granularity sentence --train_batch_size 16 --num_epochs 10 --nli_labels e 
    
    accelerate launch train_summac.py --model vitc --granularity sentence --train_batch_size 16 --num_epochs 10 --nli_labels e 
    """

    args = parser.parse_args()
    train(**args.__dict__)
