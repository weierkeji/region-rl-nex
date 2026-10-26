# Qwen3-4B 实验配置说明

## 实验目的

对比 baseline 全量权重广播与 enabled 稀疏增量广播方案的训练效果、权重传输开销和运行效率。实验分组为 `q4b-original-compare-20261023`，两组均完成 800 步训练。

## 共同配置

| 配置项 | 设置 |
|---|---|
| 模型 | Qwen3-4B，训练模型实现为 `hf` |
| 模型路径 | `/inspire/hdd/project/qianghuaxuexi/public/models/Qwen3-4B` |
| 训练环境 | `deepscaler-math`，环境名称 `deepscaler` |
| 数据路径 | `/tmp/q4b-compare-20261023/data` |
| 训练步数 | 800 |
| 学习率 | `1e-6` |
| Batch size / Group size | 512 / 8 |
| 最大并发 rollout | 512 |
| 序列长度 / 最大生成长度 | 2048 / 2048 tokens |
| 训练 GPU 数 | 2 |
| 激活检查点 | `trainer.model.ac.freq = 1` |
| Checkpoint 保存间隔 | 100 步 |
| Lease / Lease recovery | 均关闭 |
| 样本与分布记录 | 开启，间隔 10 步，样本比例 5% |

本实验未在 RL 配置中提供 `[inference]`，因此 RL 入口不启动推理服务；`deployment.num_infer_gpus = 0` 同时表示该入口不为推理分配 GPU。外层启动脚本先用 GPU 0–3 启动推理服务，等待各服务 `/health` 就绪，再用 `CUDA_VISIBLE_DEVICES=4,5` 启动 RL。Orchestrator 通过配置的 `base_url` 和 `admin_base_url` 连接这些服务。不能仅凭 `num_infer_gpus=0` 推断任何配置下都不会启动推理进程，入口实际检查的是 `config.inference`。

## 两组差异

| 配置项 | Baseline | Enabled |
|---|---|---|
| 结果目录 | `baseline` | `enabled-cpu1-20261025` |
| 广播类型 | `filesystem` | `filesystem` |
| 权重内容 | `mode = "full"` | `mode = "delta"` |
| 更新协议 | `direct` | `stage_commit` |
| 传输方式 | `shared_fs` | `streaming_upload` |
| 后台 staging | 关闭 | 开启 |
| 保留全部 delta | 关闭 | 关闭 |
| 推理服务 | 1 个 endpoint，DP=4、TP=1 | 4 个独立 endpoint，每个使用 1 张 GPU |
| 推理端口 | 8511 | 8501–8504 |
| 负载均衡 | 关闭 | 开启 |

Enabled 的推理进程显式设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。其来源记录为分支 `fix/sparse-delta-fp64-fallback`、提交 `317ece8011820417ce28d8ab7d72f8bbfec48277`，包含差值无法精确还原时的 FP64 回退。

这是两套运行方案的对比，推理拓扑、负载均衡和 CPU 线程配置也参与影响结果，不能将全部性能差异单独归因于 sparse delta。

## Enabled 核心配置

以下是关键配置节选；完整运行还需模型、数据及推理服务配置。

```toml
[weight_broadcast]
type = "filesystem"
mode = "delta"
update_protocol = "stage_commit"
stage_transport = "streaming_upload"
background_stage = true
retain_all_deltas = false

[orchestrator]
batch_size = 512
group_size = 8
max_inflight_rollouts = 512
enable_load_balancing = true
```

## 日志与结果

训练时使用 W&B 离线模式，原配置项目名为 `aurorarl`；完成后统一上传至 `chen7710_2023-fdu-knights-athletics/auroraRL`。每组包含 trainer、orchestrator 和 parameter-changes 三个 run，分别记录训练指标、rollout 指标及参数变化/传输统计。

[W&B 对比视图](https://forge.coreweave.com/wandb/chen7710_2023-fdu-knights-athletics/auroraRL/runs/fpgn5fxx) 包含 43 张叠加图：蓝线为 baseline，橙线为 enabled。Trainer/orchestrator 使用实际 `step`；参数统计使用经原始 JSONL 验证的 optimizer update 编号。

完整配置与来源位于本地归档（`outputs/` 不纳入 Git，因此以下路径不会随分支上传）：

- Baseline：`outputs/q4b-original-compare-20261023/baseline/` 下的 `rl.toml`、`inference-0.toml`。
- Enabled：`outputs/q4b-original-compare-20261023/enabled-cpu1-20261025/` 下的 `rl.toml`、`launch.sh`、`source.json`。

复现时使用上述归档配置，并确认模型、数据和启动脚本依赖的 `/tmp` 运行环境仍然存在；未列出的默认参数以对应实验的 `resolved/` 配置为准。
