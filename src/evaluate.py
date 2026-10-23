from transformers import AutoModelForCausalLM, AutoConfig
import torch, math
from src.data import TokenBlockDataset
from torch.utils.data import DataLoader
import deepspeed
from src.model import load_model, build_tiny_model
from torch.utils.data import Subset
import torch.distributed as dist

def evaluate(engine: deepspeed.DeepSpeedEngine, val_loader:DataLoader) :
    engine.eval()
    with torch.no_grad() :
        loss_sum = 0
        token_sum = 0
        iterator = iter(val_loader)
        while True :
            try :
                batch = next(iterator)
                output = engine(
                    input_ids = batch['input_ids'].to(engine.device),
                    labels = batch['labels'].to(engine.device),
                    use_cache = False
                )
                batch_size, seq_len = batch['input_ids'].shape
                n = batch_size * ( seq_len - 1 )
                loss_sum += output.loss.item() * n
                token_sum += n 

            except StopIteration :
                break
    totals = torch.tensor([loss_sum, token_sum],device=engine.device,dtype=torch.float64)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    loss_sum, num_tokens = totals.tolist()
    nll = loss_sum / num_tokens
    perplexity = math.exp(nll)
    engine.train()
    return (nll, perplexity, num_tokens)