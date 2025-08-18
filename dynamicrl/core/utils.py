import logging
import random
import numpy as np
import torch

logger = logging.getLogger(__name__)

#set the random seed for reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    logger.info(f"Global random seed set to {seed}")