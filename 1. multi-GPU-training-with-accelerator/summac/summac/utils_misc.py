import numpy as np
import requests
import tqdm
import os

from accelerate.logging import get_logger

logger = get_logger(__name__)

# GPU-related business
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi')
    memory_available = [int(x.split()[2]) + 5 * i for i, x in enumerate(open('tmp_smi', 'r').readlines())]
    os.remove("tmp_smi")
    return np.argmax(memory_available)


def select_freer_gpu():
    # freer_gpu = str(get_freer_gpu())
    freer_gpu = 0, 1
    logger.debug("Will use GPU: %s, %s" % freer_gpu, main_process_only=True)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    return freer_gpu


def batcher(iterator, batch_size=4, progress=False):
    if progress:
        iterator = tqdm.tqdm(iterator)

    batch = []
    for elem in iterator:
        batch.append(elem)
        if len(batch) == batch_size:
            final_batch = batch
            batch = []
            yield final_batch
    if len(batch) > 0:  # Leftovers
        yield batch


# Google Drive related

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
