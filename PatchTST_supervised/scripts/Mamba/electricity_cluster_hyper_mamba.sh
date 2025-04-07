export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=2

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=Mamba

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=Electricity # 如果是聚类后的模型，model_id_name后面再加上_cluster
data_name=custom

random_seed=2024

for d_model in 32 64 128 256 512
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
  --n_clusters 3 \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model $d_model \
  --d_state 16 \
  --d_ff 512 \
  --is_flip 1 \
  --dropout 0.3 \
  --des 'd_model''敏感性' \
  --use_wandb True \
  --train_epochs 10 \
  --patience 5\
  --lradj '5'\
  --gpu ${gpu} \
  --itr 1 --batch_size 128 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done

for d_ff in 32 64 128 256 512
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
  --n_clusters 3 \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 512 \
  --d_state 16 \
  --d_ff $d_ff \
  --is_flip 1 \
  --dropout 0.3 \
  --des 'd_ff''敏感性' \
  --use_wandb True \
  --train_epochs 10 \
  --patience 5\
  --lradj '5'\
  --gpu ${gpu} \
  --itr 1 --batch_size 128 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done

for d_state in 2 4 8 16 32
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
  --n_clusters 3 \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 512 \
  --d_state $d_state \
  --d_ff 512 \
  --is_flip 1 \
  --dropout 0.3 \
  --des 'd_state''敏感性' \
  --use_wandb True \
  --train_epochs 10 \
  --patience 5\
  --lradj '5'\
  --gpu ${gpu} \
  --itr 1 --batch_size 128 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done