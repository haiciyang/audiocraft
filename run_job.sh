#!/bin/bash -l
#SBATCH -J mg
#SBATCH -o slurm_output/%J.output
#SBATCH -e slurm_output/%J.output
#SBATCH --mail-user hyang@merl.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus=4                # total number of GPUs
#SBATCH --cpus-per-task=32        # CPU cores per MPI process
#SBATCH -p gpu_48
#SBATCH -w node04

source ~/.bashrc
conda activate audiocraft

cd /mm0/hyang/Projects/audiocraft


# Change master_port code when "The server socket has failed to listen on any local network address" - [torchrun --nproc_per_node=2 --master_port 12434  train.py --run_ddp .....]

if test -d /local-data/hyang/fma_large_wav_splits; then
  echo "Data Directory exists."
else 
    echo "Copying data"
    if ! test -d /local-data/hyang/; then
        mkdir -p /local-data/hyang/fma_large_wav_splits
    fi
    rsync -arh --info=progress2 /mm0/hyang/scratch/Data/fma_large_wav_splits /local-data/hyang/
fi

if test -f /local-data/hyang/fma_large_wav_splits/test/000/000181.json; then
  echo "Json files exist."
else 
  cd /mm0/hyang/Projects/audiocraft/prepare_data
  python prepare_json.py
fi

cd /mm0/hyang/Projects/audiocraft


# Encodec 24khz
# dora run -d solver=musicgen/musicgen_baseline compression_model_checkpoint=//pretrained/facebook/encodec_24khz sample_rate=24000 transformer_lm.card=1024 transformer_lm.n_q=8 +compression_model_n_q=8 codebooks_pattern.delay.delays=[0,1,2,3,4,5,6,7]

# Encodec 32khz
# dora run -d solver=musicgen/musicgen_baseline model/lm/model_scale=small compression_model_checkpoint=//pretrained/facebook/encodec_32khz sample_rate=32000 transformer_lm.card=2048 transformer_lm.n_q=4 +compression_model_n_q=4 codebooks_pattern.delay.delays=[0,1,2,3] 
# continue_from=//sig/af50f5c4

# DAC 24khz
# dora run -d solver=musicgen/musicgen_baseline compression_model_checkpoint=//pretrained/dac_24khz sample_rate=24000 transformer_lm.card=1024 transformer_lm.n_q=9 +compression_model_n_q=9 codebooks_pattern.delay.delays=[0,1,2,3,4,5,6,7,8] dataset.batch_size=32 

# DAC 44khz
# dora run -d solver=musicgen/musicgen_baseline compression_model_checkpoint=//pretrained/dac_44khz sample_rate=44100 transformer_lm.card=1024 transformer_lm.n_q=9 +compression_model_n_q=9 codebooks_pattern.delay.delays=[0,1,2,3,4,5,6,7,8] dataset.batch_size=32 

# SQCodec
# dora run -d solver=musicgen/musicgen_baseline compression_model_checkpoint=//pretrained/sqcodec sample_rate=16000 transformer_lm.card=19683 transformer_lm.n_q=4 +compression_model_n_q=4 codebooks_pattern.delay.delays=[0,1,2,3] dataset.batch_size=32 
dora run -d solver=musicgen/musicgen_baseline compression_model_checkpoint=//pretrained/sqcodec sample_rate=16000 transformer_lm.card=19683 transformer_lm.n_q=4 +compression_model_n_q=4 codebooks_pattern.delay.delays=[0,1,2,3] dataset.batch_size=32 continue_from=//sig/72d4d300
