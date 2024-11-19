conda activate rec
cd RecStudio

export CUDA_VISIBLE_DEVICES=$1

python gre.py \
    --model SASRec \
    --embed_dim 64 \
    --dataset_path $3 \
    --batch_size $4 \
    --eval_batch_size $4