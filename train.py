import argparse
import json
from src.data import TokenBlockDataset
from torch.utils.data import DataLoader
import deepspeed
from src.model import build_tiny_model
from torch.utils.data import Subset
from src.evaluate import evaluate
from pathlib import Path
import os
import logging
import uuid
from datetime import datetime
from src.random_state import seed_everything, capture_rng_state, restore_rng_state
from src.checkpoint import build_resume_config, check_resume_config, save_checkpoint
import torch.distributed as dist
from torch.utils.data import DistributedSampler

def capture_all_rng_states():
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, capture_rng_state())
    return states

def getlog() -> logging.Logger:
    log_dir = "runs/"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = uuid.uuid4().hex[:6]  # 截取 6 位随机十六进制字符
    log_filename = f"train_{timestamp}_{random_suffix}.log"
    log_path = os.path.join(log_dir, log_filename)
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
    logger.addHandler(logging.StreamHandler())
    logger.propagate = False
    return logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="单卡小模型训练与 checkpoint 恢复测试")
    parser.add_argument("--ds-config", type=Path, default=Path("configs/deepspeed.json"),
                        help="DeepSpeed JSON 配置文件路径")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/resume_test01"))
    parser.add_argument("--resume-tag", default=None, help="例如 step10；不传则从头训练")
    parser.add_argument("--max-updates", type=int, default=20, help="累计更新步数目标，包含恢复前的步数")
    # DeepSpeed launcher 会传入此参数，实际设备仍由 engine 管理。
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_updates < 0:
        parser.error("--max-updates 必须大于或等于 0")
    if args.resume_tag is not None:
        if not (args.checkpoint_dir / args.resume_tag).is_dir():
            parser.error(f"找不到 checkpoint：{args.checkpoint_dir / args.resume_tag}")
    elif args.checkpoint_dir.exists():
        if not args.checkpoint_dir.is_dir() or any(args.checkpoint_dir.iterdir()):
            parser.error("从头训练需要新的或空的 checkpoint 目录；恢复请指定 --resume-tag")
    return args


def main(argv=None):
    args = parse_args(argv)

    model_id="Qwen/Qwen2.5-1.5B"
    revision="8faed761d45a263340a0528343f099c05c9a4323"
    ds_config = json.loads(args.ds_config.read_text(encoding="utf-8"))
    data_dir = Path("data/smoke")

    seed_everything(args.seed)


    model = build_tiny_model(model_id=model_id, revision=revision)
    engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
    dataset = TokenBlockDataset.from_prepared(
       data_dir, 'train', 128
    )
    validset = TokenBlockDataset.from_prepared(
        data_dir, 'validation', 128
    )
    # 此阶段固定顺序、单进程读取；恢复时可重建相同顺序并跳过已消费的 batch。
    # loader = DataLoader(dataset, batch_size=ds_config["train_micro_batch_size_per_gpu"],
    #                     shuffle=False, num_workers=0)
    # vloader = DataLoader(Subset(validset, [0, 1, 2, 3, 4, 5, 6, 7]))

    # 加载数据
    

    # 分布式信息
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 参数
    epoch = 0
    batches_in_epoch = 0
    max_updates = args.max_updates
    evaluate_interval = 5
    save_interval = 10
    las_upd = 0
    las_save = 0
    #
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=ds_config["train_micro_batch_size_per_gpu"],
        sampler=sampler,
        num_workers=0,
    )

    indices = range(rank, min(8, len(validset)), world_size)
    vloader = DataLoader(
        Subset(validset, indices),
        batch_size=1,
        num_workers=0,
    )



    logger = getlog()
    checkpoint_dir = args.checkpoint_dir
    resume_config = build_resume_config(
        seed=args.seed, model_id=model_id, revision=revision, model=model,
        data_dir=data_dir, loader=loader, ds_config=ds_config,
    )
    if args.resume_tag is not None:
        check_resume_config(checkpoint_dir, args.resume_tag, resume_config)
        load_path, client_state = engine.load_checkpoint(str(checkpoint_dir), tag=args.resume_tag)
        las_upd = client_state['las_upd']
        las_save = client_state['las_save']
        epoch = client_state['epoch']
        batches_in_epoch = client_state['batches_in_epoch']
        if rank == 0 :
            logger.info("recover from %d: epoch=%d batches_in_epoch=%d",
                        engine.global_steps, epoch, batches_in_epoch)
            logger.info(
                "update_step=%d lr_next=%.8g",
                engine.global_steps,
                engine.get_lr()[0],
            )
    else :
        client_state = {
            "las_upd" : las_upd,
            "las_save" : las_save,
            "epoch" : epoch,
            "batches_in_epoch": batches_in_epoch,
            "rng_state": capture_all_rng_states()
        }
        
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(batches_in_epoch):
        next(iterator)
    nll, perplexity, num_tokens = evaluate(engine, vloader)
    if rank == 0:
        logger.info("%d %f %f %d", engine.global_steps, nll, perplexity, num_tokens)

    engine.train()
    restore_rng_state(client_state['rng_state'][rank])
    if args.resume_tag is None :
        save_checkpoint(engine, checkpoint_dir, "step0", client_state, resume_config)


    while engine.global_steps < max_updates:
        try :
            batch = next(iterator)
        except StopIteration :
            epoch = epoch + 1
            batches_in_epoch = 0
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)

        output = engine(
            input_ids = batch['input_ids'].to(engine.device),
            labels = batch['labels'].to(engine.device),
            use_cache = False
        )
        engine.backward(output.loss)
        engine.step()
        # 一个 batch 就加一次，与是否达到梯度累积边界无关。
        batches_in_epoch += 1
        logger.info("[rank%d] train epoch=%d sample_id=%s update_step=%d batches_in_epoch=%d",
                        rank, epoch, batch['sample_id'].tolist(), engine.global_steps, batches_in_epoch)
        if engine.global_steps != las_upd and engine.global_steps % evaluate_interval == 0 :
            las_upd = engine.global_steps
            nll, perplexity, num_tokens = evaluate(engine, vloader)
            if rank == 0:
                logger.info("%d %f %f %d", engine.global_steps, nll, perplexity, num_tokens)
                logger.info(
                    "update_step=%d lr_next=%.8g",
                    engine.global_steps,
                    engine.get_lr()[0],
                )
            #print(engine.global_steps, nll, perplexity, num_tokens)
        
        if engine.global_steps != las_save and engine.global_steps % save_interval == 0 :
            las_save = engine.global_steps
            client_state = {
                "las_upd" : las_upd,
                "las_save" : las_save,
                "epoch" : epoch,
                "batches_in_epoch": batches_in_epoch,
                "rng_state": capture_all_rng_states()
            }
            save_checkpoint(engine, checkpoint_dir, f"step{engine.global_steps}", client_state, resume_config)

    if rank == 0:
        logger.info("finished: update_step=%d epoch=%d batches_in_epoch=%d",
                    engine.global_steps, epoch, batches_in_epoch)


if __name__ == "__main__":
    main()
