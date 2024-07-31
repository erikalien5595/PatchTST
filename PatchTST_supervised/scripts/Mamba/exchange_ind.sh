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

root_path_name=./dataset/exchange_rate/
data_path_name=exchange_rate.csv
model_id_name=Exchange # 如果是聚类后的模型，model_id_name后面再加上_cluster
data_name=custom

random_seed=2024
for pred_len in 96 192 336 720
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
      --ch_ind 1 \
      --is_cluster 0 \
      --revin 1 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 8 \
      --e_layers 2 \
      --n_heads 16 \
      --d_model 128 \
      --d_state 32 \
      --d_ff 128 \
      --is_flip 1 \
      --dropout 0.1\
      --fc_dropout 0.3 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'FlipChannelIndependence' \
      --train_epochs 10 \
      --patience 3\
      --lradj '5'\
      --pct_start 0.2\
      --gpu ${gpu} \
      --itr 1 --batch_size 16 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
