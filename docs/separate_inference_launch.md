# 独立启动推理服务与训推连接

## 启动方式

本实验由外层 Shell 脚本统一管理进程：先启动推理服务，等待服务就绪，再启动 RL 入口。RL 入口负责启动 Trainer 和 Orchestrator，推理服务独立运行。

```text
外层启动脚本
├── 推理服务 0：GPU 0，端口 8501
├── 推理服务 1：GPU 1，端口 8502
├── 推理服务 2：GPU 2，端口 8503
├── 推理服务 3：GPU 3，端口 8504
└── RL 入口：可见 GPU 4、5
    ├── Trainer：执行模型训练
    └── Orchestrator：调度 rollout 和权重更新
```

四个推理服务分别加载一份 Qwen3-4B。启动脚本使用 `CUDA_VISIBLE_DEVICES` 隔离 GPU，并为每个服务指定独立配置、端口和日志文件。

下面是等价的启动命令示意，需要在已安装本项目依赖的环境执行；`$result` 指向实验配置目录：

```bash
# 在后台启动四个推理服务。
for i in 0 1 2 3; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$i" \
    uv run inference @ "$result/inference-$i.toml" \
    > "$result/logs/inference-$i.log" 2>&1 &
done

# 确认四个端口的 /health 均成功后，再启动 RL。
CUDA_VISIBLE_DEVICES=4,5 uv run rl @ "$result/rl.toml"
```

实际归档脚本使用 `/tmp/aurorarl-transfer-runtime/bin` 中的运行环境，并额外实现健康检查轮询、进程存活检查和退出清理。健康检查请求例如：

```bash
curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8501/health
```

## RL 配置

```toml
[deployment]
num_train_gpus = 2
num_infer_gpus = 0
gpus_per_node = 2
```

本实验的 RL 配置不提供 `[inference]`，因此 RL 入口不启动内置推理进程；`num_infer_gpus=0` 表示该入口不为推理分配 GPU。实际推理仍使用外层脚本分配的 GPU 0–3。是否启动内置推理的判断条件是 `config.inference`，不能仅靠 GPU 数量为零来判断。

## 服务连接

Orchestrator 通过两组 HTTP 地址使用推理服务：

```toml
[orchestrator]
enable_load_balancing = true

[orchestrator.student.client]
base_url = [
  "http://127.0.0.1:8501/v1",
  "http://127.0.0.1:8502/v1",
  "http://127.0.0.1:8503/v1",
  "http://127.0.0.1:8504/v1",
]
admin_base_url = [
  "http://127.0.0.1:8501",
  "http://127.0.0.1:8502",
  "http://127.0.0.1:8503",
  "http://127.0.0.1:8504",
]
```

| 地址 | 用途 |
|---|---|
| `base_url`，带 `/v1` | 发送生成请求，获取 rollout；启用负载均衡后分配到多个 endpoint |
| `admin_base_url`，不带 `/v1` | 上传权重文件、stage/commit，以及暂停和恢复推理 |

两组地址按相同顺序对应同一批服务。当前均使用 `127.0.0.1`，因此训练、调度和推理部署在同一台机器。跨机器部署时，需要改为 Orchestrator 可访问的推理主机地址，并保证服务监听地址、端口和管理接口访问权限正确。

## 训练与权重更新流程

```text
Orchestrator → 推理服务：发送生成请求
推理服务 → Orchestrator：返回 rollout
Orchestrator → Trainer：提供训练批次
Trainer：更新模型，生成权重广播文件
Orchestrator → 推理服务：上传并提交新权重
```

本次 enabled 使用 `filesystem + delta + stage_commit + streaming_upload`：Trainer 写入 `delta.stream`；Orchestrator 可以在文件生成期间上传已有字节，完成后暂停推理、提交更新，再恢复服务。这里的 `filesystem` 表示训练端生成文件形式的产物，HTTP 上传不要求训练端与推理端共享文件系统。

模型初始化时，训推两端必须加载一致的基座；后续 delta 必须按版本顺序应用。训练结束后，外层脚本负责清理它启动的推理及其他后台进程。

## 实验文件

本地归档目录为 `outputs/q4b-original-compare-20261023/enabled-cpu1-20261025/`，包含实际启动脚本 `launch.sh`、RL 配置 `rl.toml` 和推理配置 `inference-0.toml` 至 `inference-3.toml`。`outputs/` 不纳入 Git，这些运行文件不会随分支上传。

仓库内的配置说明见 [实验配置总览](q4b_experiment_config.md)。
