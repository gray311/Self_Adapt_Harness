#!/usr/bin/env bash
#SBATCH --job-name=rri16-executor
#SBATCH --account=av_alpamayo_reasoning
#SBATCH --partition=cpu
#SBATCH --qos=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.err

set -euo pipefail
umask 027
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
: "${RR_RUN_CONFIRMED:?set RR_RUN_CONFIRMED=YES}"
exec bash "$CODE_ROOT/self_adapt_harness/scripts/drive_reward_route_inference16_executor.sh"
