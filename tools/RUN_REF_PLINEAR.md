# 无人值守正式 P-linear

同步 `run_ref_plinear.py` 和 `test_run_ref_plinear.py` 到服务器tools目录。
不修改任何训练、候选、Ref推理代码。先前通过smoke的输入、checkpoint和脚本必须仍在原位置。

## 启动

```bash
cd /media/data6/chengz/WeDetect
conda activate wedetect_ref
python -B tools/test_run_ref_plinear.py
```

测试中的 `FAILED at train` 是故意让玩具子进程以7退出的失败路径测试；最终应打印PASS。
测试不训练模型、不使用GPU。

使用当前已分配的GPU；若调度器设置CUDA_VISIBLE_DEVICES，不要覆盖。
只在手动确认GPU0可用时执行 `export CUDA_VISIBLE_DEVICES=0`。

```bash
PL_LAUNCH_LOG="ref_plinear_launch_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup python -u -B tools/run_ref_plinear.py > "$PL_LAUNCH_LOG" 2>&1 < /dev/null &
PL_RUN_PID=$!
echo "PID=$PL_RUN_PID  LOG=$PL_LAUNCH_LOG"
tail -n 25 "$PL_LAUNCH_LOG"
```

脚本使用当前conda的Python启动所有子进程。新结果目录自动命名为：

```text
results/ref_plinear_refcocog_uni_full_时间戳_PID/
results/ref_plinear_refcocog_uni_full_时间戳_PID_runner/
```

启动日志会明确打印两个绝对路径。若确需自定义新目录，可在启动前设置：

```bash
export PL_RUN_OUT=results/ref_plinear_refcocog_uni_full_custom
```

此目录及对应_runner目录都不能已存在；不要指向之前smoke目录。
脚本不继承旧PL_OUT作为输出路径，避免误用之前的smoke输出。

## 行为

1. 检查所需路径、Uni完成标记、至少25GiB可用磁盘以及CUDA/BF16支持。
2. 使用原固定数据路径和划分，强制5000train/1000dev/2573validation，清除遗留smoke数量设置。
3. 顺序执行preflight、cache、train、cache_val、evaluate。
4. 子进程退出码为0且阶段完成文件检查通过，才执行下一步。
5. 每阶段独立进程退出，释放其GPU上下文；runner自身不加载PyTorch/大模型。
6. 失败就停止、不重试、不自动换卡、不改变精度/分辨率/学习率，也不从旧目录恢复。

runner固定使用此前通过smoke的默认数据与模型路径；忽略旧shell里的PL_*路径设置，
但CUDA_VISIBLE_DEVICES原样继承。训练仍是原脚本中的五层、三seed、20epochs；不重新选协议。

## 查看进度与完成情况

```bash
tail -f "$PL_LAUNCH_LOG"
```

`Ctrl+C`只退出tail，不停止nohup任务。重新登录后，使用之前打印的日志文件名。
也可以查看打印出的 `_runner` 目录：

```text
status.json       # RUNNING / FAILED / INTERRUPTED / SUCCESS，当前阶段、PID、已完成阶段
pipeline.log      # 汇总日志
startup.log
preflight.log
cache.log
train.log
cache_val.log
evaluate.log
SUCCESS.json      # 只有五阶段全部通过才生成
```

发生失败时查看status.json中的current_stage和对应日志；不要只看Python进程消失就认为成功。
服务器进程收到SIGTERM时，runner会尝试终止自己的当前子进程，不杀其他GPU任务。
`kill -9`、服务器重启或节点被回收无法保证写入最终状态。

`nohup`用于普通SSH断开后的后台运行，不能超越集群调度时限、GPU使用授权或会话清理策略。
应在已获分配的计算节点/作业内启动，不要在登录节点抢占GPU。
当前完整Ref缓存/训练脚本无断点续跑；若失败，保留目录用于诊断，修复后新开一轮。

离开前先看启动日志，确认CUDA检查和preflight通过、已经开始cache。
本轮4080已通过smoke，但全量长序列仍可能OOM；脚本会记录失败，不会擅自降配置。
