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
from src.config import load_config, validate_run_config
import torch.profiler
# Profiler 接入：复杂的启停、导出和阶段标记封装到独立文件。
from src.profiling import TrainingProfiler

# >>> 显存监控新增：轻量计时依赖
import time
# <<< 显存监控新增


# >>> 显存监控新增：记录工具，仅用于观察，不同步 GPU 或清空缓存
class MemoryRecorder:
    def __init__(self):
        self.device = None
        self.path = None
        self.started_at = time.perf_counter()
        if not torch.cuda.is_available():
            return  # CPU 测试跳过显存采集。

        self.rank = int(os.environ.get("RANK", "0"))
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        # 在 DeepSpeed 初始化前选择该进程的卡，避免多卡时采到其他卡。
        torch.cuda.set_device(self.device)
        log_dir = Path("runs")
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.path = log_dir / f"memory_{run_id}_rank{self.rank}.jsonl"

    def begin(self):
        if self.device is not None:
            # 只重置峰值计数；开始下一区间前，先用 record 保存上一区间。
            torch.cuda.reset_peak_memory_stats(self.device)

    def record(self, stage, update_step, micro_step_in_run=0):
        if self.device is None:
            return
        gib = 1024 ** 3
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s": time.perf_counter() - self.started_at,
            "rank": self.rank,
            "device": str(self.device),
            "stage": stage,
            "update_step": update_step,
            # micro-step 从本次进程启动计数；恢复后 update_step 仍接着旧值。
            "micro_step_in_run": micro_step_in_run,
            "allocated_gib": torch.cuda.memory_allocated(self.device) / gib,
            "reserved_gib": torch.cuda.memory_reserved(self.device) / gib,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / gib,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / gib,
        }
        # 每个区间追加一行，已完成区间的记录在训练异常退出后仍可查看。
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
# <<< 显存监控新增


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


def main():
    run_config = load_config(Path("configs/train.toml"))
    ds_config = json.loads(Path("configs/deepspeed.json").read_text(encoding="utf-8"))
    validate_run_config(run_config, ds_config)

    # [model]
    model_id = run_config['model']['id']
    model_revision = run_config['model']['revision']

    # [data]
    data_dir = run_config['data']['dir']
    data_seq_len = run_config['data']['seq_len']

    # [train]
    seed = run_config['train']['seed']
    max_updates = run_config['train']['max_updates']
    # 日志频率修改：单位为参数更新步，由 TOML 控制。
    log_interval = run_config['train']['log_interval']
    evaluate_interval = run_config['train']['evaluate_interval']
    save_interval = run_config['train']['save_interval']
    checkpoint_dir = Path(run_config['checkpoint']['dir'])
    resume_tag = run_config['checkpoint']['resume_tag']


    seed_everything(seed)


    # >>> 显存监控新增：初始化区间，覆盖模型构建和 DeepSpeed 初始化
    memory_recorder = MemoryRecorder()
    memory_recorder.begin()
    # <<< 显存监控新增
    model = build_tiny_model(
        model_id=model_id,
        revision=model_revision,
        fused_loss=run_config["model"].get("fused_loss", False),
    )
    model.config.chunk_mem_const = run_config["model"].get("chunk_mem_const", 1)
    engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
    # >>> 显存监控新增：先保存初始化峰值，再进入其他阶段
    memory_recorder.record("initialize", engine.global_steps)
    # <<< 显存监控新增
    dataset = TokenBlockDataset.from_prepared(
       data_dir, 'train', data_seq_len
    )
    validset = TokenBlockDataset.from_prepared(
        data_dir, 'validation', data_seq_len
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
    las_upd = 0
    las_save = 0
    #
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=seed,
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

    # 加载断点

    logger = getlog()
    # >>> 显存监控新增：在原训练日志中给出独立显存日志的位置
    if memory_recorder.path is not None:
        logger.info("memory_log=%s", memory_recorder.path)
    # <<< 显存监控新增
    resume_config = build_resume_config(
        seed=seed, model_id=model_id, revision=model_revision, model=model,
        data_dir=data_dir, loader=loader, ds_config=ds_config,
    )
    if resume_tag:
        check_resume_config(checkpoint_dir, resume_tag, resume_config)
        # >>> 显存监控新增：单独统计 checkpoint 加载
        memory_recorder.begin()
        # <<< 显存监控新增
        load_path, client_state = engine.load_checkpoint(str(checkpoint_dir), tag=resume_tag)
        # >>> 显存监控新增：记录加载后的累计更新步数和区间峰值
        memory_recorder.record("load_checkpoint", engine.global_steps)
        # <<< 显存监控新增
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
    # >>> 显存监控新增：训练开始前的评估也单独统计
    memory_recorder.begin()
    # <<< 显存监控新增
    nll, perplexity, num_tokens = evaluate(engine, vloader)
    # >>> 显存监控新增：保存初始评估的显存记录
    memory_recorder.record("evaluate", engine.global_steps)
    # <<< 显存监控新增
    if rank == 0:
        logger.info("%d %f %f %d", engine.global_steps, nll, perplexity, num_tokens)

    engine.train()
    restore_rng_state(client_state['rng_state'][rank])
    if not resume_tag:
        # >>> 显存监控新增：step0 保存区间
        memory_recorder.begin()
        # <<< 显存监控新增
        save_checkpoint(engine, checkpoint_dir, "step0", client_state, resume_config)
        # >>> 显存监控新增：保存 step0 峰值
        memory_recorder.record("save_checkpoint", engine.global_steps)
        # <<< 显存监控新增

    # >>> 显存监控新增：独立计数，不改变原有训练与恢复状态
    memory_micro_step = 0
    memory_update_step = None
    # <<< 显存监控新增
    # 日志频率修改：避免在同一更新步的多个累积 micro-step 重复输出。
    last_logged_update = engine.global_steps
    # >>> Profiler 接入：异常退出也会关闭 Profiler；关闭开关时为无操作上下文。
    with TrainingProfiler(run_config, ds_config, rank=rank, device=engine.device, logger=logger) as profiler:
        while engine.global_steps < max_updates:
            # >>> 显存监控新增：只在完整更新周期开始时重置，保留各累积步的峰值
            if memory_update_step is None:
                memory_recorder.begin()
                memory_update_step = engine.global_steps
            # <<< 显存监控新增
            # Profiler 接入：单独标记数据读取。
            with profiler.section("train/data"):
                try :
                    batch = next(iterator)
                except StopIteration :
                    epoch = epoch + 1
                    batches_in_epoch = 0
                    sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)

            # Profiler 接入：分别标记搬运、前向、反向和参数更新。
            with profiler.section("train/h2d"):
                input_ids = batch['input_ids'].to(engine.device)
                labels = batch['labels'].to(engine.device)
            with profiler.section("train/forward"):
                output = engine(
                    input_ids=input_ids,
                    labels=labels,
                    use_cache=False,
                    chunk_mem_const=model.config.chunk_mem_const,
                )
            # 与原来的内联 .to() 一样，前向结束后释放这两个临时 Python 引用。
            del input_ids, labels
            with profiler.section("train/backward"):
                engine.backward(output.loss)
            with profiler.engine_step(engine):
                engine.step()
            # >>> 显存监控新增：参数更新完成后记录；评估/保存不会混入训练峰值
            memory_micro_step += 1
            if engine.global_steps != memory_update_step:
                memory_recorder.record("train", engine.global_steps, memory_micro_step)
                memory_update_step = None
            # <<< 显存监控新增


            batches_in_epoch += 1
            # 日志频率修改：仅 rank 0 按间隔输出，最后一个更新步也输出。
            if rank == 0 and engine.global_steps != last_logged_update and (
                engine.global_steps % log_interval == 0 or engine.global_steps == max_updates
            ):
                last_logged_update = engine.global_steps
                logger.info("[rank%d] train epoch=%d sample_id=%s update_step=%d batches_in_epoch=%d",
                            rank, epoch, batch['sample_id'].tolist(), engine.global_steps, batches_in_epoch)
            if engine.global_steps != las_upd and engine.global_steps % evaluate_interval == 0 :
                las_upd = engine.global_steps
                # >>> 显存监控新增：周期评估区间
                memory_recorder.begin()
                # <<< 显存监控新增
                # Profiler 接入：若采样窗口覆盖评估，则标明该阶段。
                with profiler.section("train/evaluate"):
                    nll, perplexity, num_tokens = evaluate(engine, vloader)
                # >>> 显存监控新增：记录周期评估峰值
                memory_recorder.record("evaluate", engine.global_steps, memory_micro_step)
                # <<< 显存监控新增
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
                # >>> 显存监控新增：保存区间包含 RNG 状态收集
                memory_recorder.begin()
                # <<< 显存监控新增
                # Profiler 接入：若采样窗口覆盖保存，则标明该阶段。
                with profiler.section("train/save_checkpoint"):
                    client_state = {
                        "las_upd" : las_upd,
                        "las_save" : las_save,
                        "epoch" : epoch,
                        "batches_in_epoch": batches_in_epoch,
                        "rng_state": capture_all_rng_states()
                    }
                    save_checkpoint(engine, checkpoint_dir, f"step{engine.global_steps}", client_state, resume_config)
                # >>> 显存监控新增：记录周期保存峰值
                memory_recorder.record("save_checkpoint", engine.global_steps, memory_micro_step)
                # <<< 显存监控新增

            # Profiler 接入：每个 micro-step 完整结束后推进一次，退出 with 时自动停止。
            profiler.step()
    # <<< Profiler 接入

    if rank == 0:
        logger.info("finished: update_step=%d epoch=%d batches_in_epoch=%d",
                    engine.global_steps, epoch, batches_in_epoch)


if __name__ == "__main__":
    main()
