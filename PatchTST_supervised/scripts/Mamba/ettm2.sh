export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=Mamba

root_path_name=./dataset/ETT-small/
data_path_name=ETTm2.csv
model_id_name=ETTm2 # 如果是聚类后的模型，model_id_name后面再加上_cluster
data_name=ETTm2

random_seed=2024
for ch_ind in 1 0
do
  if [ $ch_ind == 0 ]
    then ch_ind_name='ChannelMixing'
  else
    ch_ind_name='ChannelIndependence'
  fi
  echo 'ch_ind='$ch_ind', which means '$ch_ind_name
  for pred_len in 96 192 336 720
  do
    for random_seed in 2023 2024 42 1107 2025
    do
      python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --ch_ind $ch_ind \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --is_cluster 0 \
        --n_clusters 2 \
        --revin 0 \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers 3 \
        --n_heads 16 \
        --d_model 512 \
        --d_state 16 \
        --is_flip 1 \
        --d_ff 512 \
        --dropout 0.1\
        --fc_dropout 0.1 \
        --head_dropout 0 \
        --patch_len 16 \
        --stride 8 \
        --des 'Flip_'$ch_ind_name'_BL1_Seed'$random_seed \
        --train_epochs 30 \
        --patience 5\
        --lradj 'type3'\
        --pct_start 0.2\
        --gpu ${gpu} \
        --itr 1 --batch_size 512 --learning_rate 0.0005 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
    done
  done
done