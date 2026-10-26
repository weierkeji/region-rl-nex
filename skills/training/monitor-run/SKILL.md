---
name: monitor-run
description: Monitor an ongoing prime-rl training run — find the output directory, tail logs, check key metrics, inspect SLURM jobs, and restart safely. Use when asked to check on a run, debug training, or investigate performance.
---

# Monitor a run

## Runbook

### On launch

1. Find the output dir and read the resolved configs at `{output_dir}/configs/` (start with `rl.toml`).
2. Confirm all processes are alive and the run is making progress.
3. Write the initial summary into `{output_dir}/STATUS.md`.

### Recurring check-ins

Default cadence: **1 hour** (researcher can override). At each check-in:

1. Confirm processes are alive.
2. Grep logs for errors/warnings; note current step and key metrics.
3. **Append** an entry to `{output_dir}/STATUS.md` (never overwrite):

```markdown
## YYYY-MM-DD HH:MM UTC

**Step**: {current_step} / {max_steps}
**Health**: {Healthy | Degraded | Down}

**Progress**: reward/mean, seq_len, truncation, eval scores, env-specific metrics.
**Stability**: entropy, mismatch_kl, grad_norm — flag spikes.
**Performance**: trainer vs orchestrator step time, env lag, inference pressure.

**Notes**: anything unusual (errors, restarts, hangs). Omit if nothing notable.
```

### Restarting a run

**Never restart unless the researcher explicitly asked.** Confirm the exact restart command and the conditions that warrant one.

**Never** run kill or launch commands from your own shell. Dispatch them to the tmux **Launcher** window so the researcher sees what was executed:

```bash
SESSION=$(tmux display-message -p '#S')
tmux send-keys -t "$SESSION:Launcher" 'your command here' Enter
```

After a restart, verify all processes are back up and progress resumed before the next check-in.

---

## Reference

### Where to find things

- `scripts/tmux.sh` launches the run with a `Launcher` window in the named tmux session. The Claude window receives the output dir and session name in its appended prompt — if either is missing, **ask** rather than guess.
- `{output_dir}/configs/` — resolved TOMLs (`rl.toml` has the full picture).
- `{output_dir}/logs/` — see below.
- `{output_dir}/rollouts/step_N/` — saved rollouts.

### Logs

```
{output_dir}/logs/
├── trainer.log                # rank 0 stdout
├── orchestrator.log           # orchestrator stdout
├── inference.log              # vLLM stdout
├── trainer/
│   ├── node_*.log             # per-node (multi-node only)
│   └── torchrun/              # per-rank stdout/stderr
├── inference/
│   ├── node_*.log             # per-node (multi-node only)
│   └── router_0.log           # vllm-router per replica (multi-node only)
└── envs/{train,eval}/{env_name}/
    ├── env_server.log
    └── env_worker_*.log
```

Usually tailing `trainer.log`, `orchestrator.log`, and `inference.log` is enough. Drop into per-node or per-rank logs only when debugging. All logs are loguru with `HH:mm:ss  LEVEL  message`; levels: `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`.

Scan for problems:

```bash
grep -E "WARNING|ERROR" {output_dir}/logs/{trainer,orchestrator,inference}.log
grep -E "WARNING|ERROR" {output_dir}/logs/envs/train/*/env_{server,worker_*}.log
```

### Metrics

All metrics print to the console log (and W&B when configured).

**Progress** — orchestrator log:

| Metric | Description |
|--------|-------------|
| `reward/{all,env}/mean` | mean training reward |
| `seq_len/{all,env}/mean` | avg sequence length (tokens) |
| `num_turns/{all,env}/mean` | avg turns per rollout (multi-turn only) |
| `is_truncated/{all,env}/mean` | fraction truncated |
| `empty_rollouts/{all,env}`, `errored_rollouts/{all,env}` | fraction empty/errored |
| `metrics/{env}/{metric}` | env-specific (e.g. pass rate) |
| `eval/{env}/{avg@k,pass@k}` | eval scores when configured |

**Stability** — trainer log:

| Metric | Description |
|--------|-------------|
| `mismatch_kl/{all,env}/{mean,std,max}` | KL between trainer and (old) inference policy over trainable tokens |
| `entropy/{all,env}/{mean,std,max}` | policy entropy over trainable tokens |
| `masked_advantage_{positive,negative}/mean` | fraction of DPPO-masked tokens with +/- advantage |
| `optim/grad_norm` | spikes may precede divergence |

**Performance** — trainer and orchestrator step independently, so comparing step times shows who's waiting on whom.

| Source | Metric | Description |
|--------|--------|-------------|
| trainer | `time/step` | total trainer step |
| trainer | `time/wait_for_batch` | **high → orchestrator is bottleneck** |
| trainer | `time/forward_backward`, `time/broadcast_weights`, `time/save_ckpt` | phase timings |
| trainer | `perf/throughput`, `perf/mfu` | tokens/s and MFU % |
| orchestrator | `time/step`, `time/generate_completions`, `time/update_weights` | phase timings |
| orchestrator | `time/wait_for_ckpt` | **high → trainer is bottleneck** |
| orchestrator | `scheduler/async_level`, `scheduler/inflight_rollouts` | scheduler state |
| env server | event loop lag (min/mean/p90/p99/max), active task distribution | periodic |

For live vLLM stats, query Prometheus directly:

```bash
curl -s http://localhost:8000/metrics | grep -E "num_requests|gpu_cache_usage"
# vllm:num_requests_running, vllm:num_requests_waiting, vllm:gpu_cache_usage_perc (→1.0 = KV cache saturated)
```

### Uploading offline W&B runs

When asked to upload an archived run, inspect both `wandb/` and `orchestrator-wandb/` for `offline-run-*/run-*.wandb`; auxiliary statistics may have their own run. Pass explicit run directories to `uv run wandb sync --entity <entity> --project <project> <run-directory> ...` to preserve run IDs and groups without uploading unrelated experiments. Use the exact destination project spelling, which can differ from the archived config.

If sync reports no API key, have the user run `uv run wandb login` in the repository terminal; never request or print the key in chat. A sandbox `ProxyError` requires rerunning with network escalation, not changing credentials or disabling the proxy. Wait for each sync to finish and verify the remote run IDs and history before reporting success. Sync uploads the recorded W&B data and associated files, not arbitrary CSV/JSON files elsewhere in the archive.

`wandb sync` can exit successfully while artifact uploads fail. Archived incremental tables may reference missing `/root/.local/share/wandb/artifacts/staging/tmp*` files even when `files/media/table/*.table.json` survives. With SDK 0.26.1, recover these using the original SDK: scan records with `DataStore`, match history `samples.artifact_path` to each artifact client ID and manifest path, rebuild `wandb.Table` from the archived columns/data, restore `increment_num` and `previous_increments_paths`, and serialize its artifact JSON with `json.dumps(..., sort_keys=True)`. Verify every payload against the original manifest's base64 MD5 and byte length before use; stop if either differs. Rewrite only manifest `local_path` fields in a separate `.wandb` copy using `DataStore.write`, copy its associated files, and sync that copy to the same run ID. Preserve original records, artifact IDs, digests and history; never substitute plain media JSON for the richer artifact JSON. Verify remote committed artifact versions as well as metrics after recovery.

If the repaired incremental artifacts still fail with `invalid artifact manifest digest ... expected , got ...`, do not report full artifact recovery. A verified fallback is to resume the same remote run with `resume="must"` in online mode, concatenate the recovered increments in increment order, and log an ordinary immutable `wandb.Table` under a separate `samples_recovered` key. Keep the original metrics/history and explain that old incremental-table references remain unresolved. Confirm the new artifact is `COMMITTED`, download it, and compare every row and column against the recovered input. Record the recovery count and reason in run summary; this adds one history record and does not represent another training step.

### Comparing archived runs in W&B

For a comparison view without changing original runs, extract matching scalar metrics from each role's offline records and log paired `wandb.Plotly` line charts in a separate run with `job_type="comparison"`. Use fixed colors and legend names across charts, and record source run IDs and aggregation rules. Trainer/orchestrator curves use the explicit `step` field, not W&B `_step`, because logging frequencies differ. Parameter-change observers use `_step` as the optimizer update; verify this against `parameter_changes.jsonl` before plotting. When multiple observations share a step, explicitly document the reduction (for example, last finite observation per metric); do not silently smooth or align by wall time. Distinguish trainer and orchestrator metrics with identical names by panel prefixes. Verify uploaded Plotly JSON has both series, expected colors and exact source data. A comparison run is a visualization, not another training experiment.

### Rollouts

```
{output_dir}/rollouts/step_N/
├── train_rollouts.jsonl   # all train rollouts (vf.RolloutOutput, trajectory excluded)
├── eval_rollouts.jsonl    # only present when eval ran
└── train_rollouts.bin     # binary batch consumed by the trainer
```

```bash
wc -l {output_dir}/rollouts/step_42/train_rollouts.jsonl
head -1 {output_dir}/rollouts/step_42/train_rollouts.jsonl | python -m json.tool
jq '.reward' {output_dir}/rollouts/step_42/train_rollouts.jsonl
```

### Common failure modes

A few warnings are normal. Escalate when errors are persistent, growing, or hit a large fraction of rollouts.

- **Env workers**: exceptions in env code, timeouts, sandbox errors, OOM kills (most common source — runs user code).
- **Orchestrator**: empty/errored rollout spikes, weight-broadcast failures, checkpoint errors.
- **Trainer**: NCCL/CUDA errors, OOM, NaN loss or gradients.
- **Inference**: NCCL/CUDA errors, OOM, request timeouts.

### Process tree

All processes use `setproctitle` so they're visible in `ps`/`htop`/`pstree`:

```
PRIME-RL::Launcher
├── PRIME-RL::Inference          (vLLM server, GPU 0)
├── PRIME-RL::Orchestrator       (CPU-only)
│   └── Verifiers::EnvServer     (ZMQ env server per environment)
│       └── Verifiers::EnvWorker0..N
├── torchrun
│   └── PRIME-RL::Trainer        (GPU 1+)
└── tail trainer.log
```

For multi-node runs, trainer and inference processes are on separate nodes — use `srun` or `ssh` to inspect them.
