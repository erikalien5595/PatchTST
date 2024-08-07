export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=Mamba

root_path_name=./dataset/PEMS/
data_path_name=PEMS07.npz
model_id_name=PEMS07 # 如果是聚类后的模型，model_id_name后面再加上_cluster
data_name=PEMS

random_seed=2021
for pred_len in 12 24 48 96
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --is_cluster 0 \
      --n_clusters 3 \
      --revin 1 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 883 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 512 \
      --d_state 16 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Flip' \
      --train_epochs 10 \
      --patience 10\
      --lradj '5'\
      --pct_start 0.2\
      --gpu ${gpu} \
      --itr 1 --batch_size 32 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done