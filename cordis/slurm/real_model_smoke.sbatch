#!/usr/bin/env bash
# Launch one Qwen3.5-9B vLLM replica and exercise it through Cordis/DSH.
# Submit from anywhere with: sbatch cordis/slurm/real_model_smoke.sbatch
#SBATCH --job-name=sah-cordis-smoke
#SBATCH --account=av_alpamayo_reasoning
#SBATCH --partition=pool0
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/yingzim/code/sah_corids/.runtime/cordis/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/yingzim/code/sah_corids/.runtime/cordis/%x-%j.err

set -euo pipefail
umask 027
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh

REPO="/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/yingzim/code/sah_corids"
IMAGE="${SAH_IMAGE}"

srun --nodes=1 --ntasks=1 --cpus-per-task=24 --export=ALL \
  --container-image="$IMAGE" \
  --container-mounts="/lustre:/lustre,/tmp:/tmp" \
  --container-workdir="$REPO" \
  bash "$REPO/cordis/slurm/real_model_smoke_worker.sh"
