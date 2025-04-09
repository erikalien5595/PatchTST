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

root_path_name=./dataset/weather/
data_path_name=weather.csv
model_id_name=Weather # 如果是聚类后的模型，model_id_name后面再加上_cluster
data_name=custom

random_seed=2024
for corr_threshold in 0.1 0.3 0.5 0.7 0.9
do
for pred_len in 96
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
  --is_cluster 1 \
  --n_clusters 8 \
  --corr_threshold $corr_threshold \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --e_layers 2 \
  --n_heads 16 \
  --d_model 512 \
  --d_state 16 \
  --d_ff 256 \
  --is_flip 1 \
  --dropout 0.1 \
  --des 'tau敏感性' \
  --use_wandb True \
  --train_epochs 10 \
  --patience 5\
  --lradj '5'\
  --gpu ${gpu} \
  --itr 1 --batch_size 128 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done
