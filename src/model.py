from transformers import AutoModelForCausalLM, AutoConfig
import torch
from src.data import TokenBlockDataset
from torch.utils.data import DataLoader
import deepspeed

def load_model(model_id, revision) :
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
    )
    return model

def build_tiny_model(model_id, revision) :
    config = AutoConfig.from_pretrained(
        model_id,
        revision = revision,
    )
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.use_cache = False

    return AutoModelForCausalLM.from_config(config)

if __name__ == "__main__":


    model = load_model(
        model_id="Qwen/Qwen2.5-1.5B",
        revision="8faed761d45a263340a0528343f099c05c9a4323",
    )
    print(type(model).__name__)

    dataset = TokenBlockDataset.from_prepared (
        "data/smoke", "train", seq_len=128
    )
    loader = DataLoader(dataset, batch_size=1)
    iterator = iter(loader)

    tiny_model = build_tiny_model(
        model_id="Qwen/Qwen2.5-1.5B",
        revision="8faed761d45a263340a0528343f099c05c9a4323",
    )
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
            },
        },
    }
    engine, optimizer, _, _ = deepspeed.initialize(model=tiny_model, model_parameters=tiny_model.parameters(), config=ds_config)

    engine.train()
    for i in range(2) :
        batch = next(iterator)
        output = engine( 
            input_ids = batch["input_ids"].to(engine.device),
            labels = batch["labels"].to(engine.device),
            use_cache = False
        )
        engine.backward(output.loss)
        engine.step()
        print(output.loss.item(), engine.global_steps,batch["sample_id"])

    # tiny_model.eval()
    # with torch.no_grad() :
    #     output = tiny_model (
    #         input_ids = batch["input_ids"],
    #         labels = batch["labels"],
    #         use_cache = False
    #     ) 
    # print("loss: ", output.loss.item())
    # print("param: ",sum(p.numel() for p in tiny_model.parameters()))







    # model.eval()
    # with torch.no_grad() :
    #     output = model (
    #         input_ids = batch["input_ids"],
    #         labels = batch["labels"],
    #         use_cache = False,
    #     )
    # print(output.loss.item())