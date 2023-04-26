from torch.optim import SGD, AdamW
from model_summac import SummaCConv
from accelerate.logging import get_logger
logger = get_logger(__name__)

def build_optimizer(model: SummaCConv, optimizer_name="adam", learning_rate=1e-5):
    logger.info(f"build optimizer={optimizer_name} on model={type(model)}", main_process_only=True)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if optimizer_name == "adam":
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = SGD(optimizer_grouped_parameters, lr=learning_rate)
    else:
        assert False, "optimizer_name = '%s' is not `adam` or `lamb`" % (optimizer_name)
    return optimizer
