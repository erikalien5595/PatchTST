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

for learning_rate in 0.00001 0.00005 0.0001 0.001 0.01
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
  --des 'learning_rate''敏感性' \
  --use_wandb True \
  --train_epochs 20 \
  --patience 10 \
  --lradj 'type3' \
  --gpu ${gpu} \
  --itr 1 --batch_size 512 --learning_rate $learning_rate #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done

for dropout in 0.0 0.1 0.2 0.3 0.4
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
  --dropout $dropout \
  --des 'dropout''敏感性' \
  --use_wandb True \
  --train_epochs 20 \
  --patience 10 \
  --lradj 'type3' \
  --gpu ${gpu} \
  --itr 1 --batch_size 512 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done

for batch_size in 32 64 128 256
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
  --des 'batch_size''敏感性' \
  --use_wandb True \
  --train_epochs 20 \
  --patience 10 \
  --lradj 'type3' \
  --gpu ${gpu} \
  --itr 1 --batch_size $batch_size --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done
