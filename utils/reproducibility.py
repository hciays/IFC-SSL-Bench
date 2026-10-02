import torch
import os
import numpy as np
import random
import lightning as L


def set_deterministic(
        enable: bool = True, seed: int = 0, cublas_ws_config: str = ":4096:8", fill_uninitialized_memory: bool = True
) -> None:
    """Reduce randomness and enable the most deterministic possible execution of this library.

    In this method, all RNGs used are seeded and CUDA is forced to perform deterministic operations.
    The latter can affect performance.
    WARNING: Even if this method tries to remove all known randomness,
    it can still happen that randomness occurs when using external libraries or certain torch operations
    like RNN or LSTMs (see: https://pytorch.org/docs/stable/notes/randomness.html#cuda-rnn-and-lstm).

    Args:
        enable (bool, optional): If the deterministic configuration should be set. Defaults to 'True'.
        seed (int, optional): The seed to use. Defaults to '0'.
        cublas_ws_config (str, optional): Configuration for 'CUBLAS_WORKSPACE_CONFIG' env variable.
            Necessary if you are using CUDA tensor and a CUDA Version of >10.2. Default: ':4096:8',
            which increases the library footprint in the GPU memory (~24MiB), but has a higher performance compared
            to the alternativ configuration ':16:8'.
            For more information see: https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility.
            fill_uninitialized_memory (bool, optional):
        fill_uninitialized_memory (bool, optional): Torch operations such as torch.empty() can return uninitialized
            memory, which can lead to nondeterministic operations. To avoid this, this memory can be set with known
            values (e.g. NaN is used for FP data types).
            However, this affects performance and should therefore be avoided. In general, functions that produce such
            uninitialized memory should be avoided, which is why a warning is issued when this flag is set to 'True'!
            Default: 'True'.

    """
    if enable:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        L.seed_everything(seed, workers=True, verbose=False)

        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(mode=True)
        torch.utils.deterministic.fill_uninitialized_memory = fill_uninitialized_memory

        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_ws_config

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        # if torch.xpu.is_available():
        #     torch.xpu.manual_seed(seed)

        # # needs a special PyTorch build provided by Cambricon
        # if torch.mlu.is_available():
        #    torch.mlu.manual_seed(seed)

        # if torch.npu.is_available():
        #    torch.npu.manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seeding a Dataloader worker to reproducible results.

    Each worker will have its own seed, related to the worker's id.

    Args:
        worker_id: The id of the dataloader worker.

    """
    worker_seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
