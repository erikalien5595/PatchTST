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

root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2 # 如果是聚类后的模型，model_id_name后面再加上_cluster
data_name=ETTh2

random_seed=2024

for n_clusters in 2 3 4 5
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
  --n_clusters $n_clusters \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 512 \
  --d_state 16 \
  --d_ff 512 \
  --is_flip 1 \
  --dropout 0.1\
  --fc_dropout 0.1 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des '聚类个数k敏感性' \
  --use_wandb True \
  --train_epochs 30 \
  --patience 5 \
  --lradj 'type3'\
  --pct_start 0.2\
  --gpu ${gpu} \
  --itr 1 --batch_size 256 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done

for e_layers in 1 2 3 4
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
  --n_clusters 2 \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers $e_layers \
  --n_heads 16 \
  --d_model 512 \
  --d_state 16 \
  --d_ff 512 \
  --is_flip 1 \
  --dropout 0.1\
  --fc_dropout 0.1 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'e_layers''敏感性' \
  --use_wandb True \
  --train_epochs 30 \
  --patience 5 \
  --lradj 'type3'\
  --pct_start 0.2\
  --gpu ${gpu} \
  --itr 1 --batch_size 256 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done

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
  --n_clusters 2 \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 512 \
  --d_state 16 \
  --d_ff 512 \
  --is_flip 1 \
  --dropout 0.1\
  --fc_dropout 0.1 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'learning_rate''敏感性' \
  --use_wandb True \
  --train_epochs 30 \
  --patience 5 \
  --lradj 'type3'\
  --pct_start 0.2\
  --gpu ${gpu} \
  --itr 1 --batch_size 256 --learning_rate $learning_rate #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
  --n_clusters 2 \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 512 \
  --d_state 16 \
  --d_ff 512 \
  --is_flip 1 \
  --dropout $dropout \
  --fc_dropout 0.1 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'dropout''敏感性' \
  --use_wandb True \
  --train_epochs 30 \
  --patience 5 \
  --lradj 'type3'\
  --pct_start 0.2\
  --gpu ${gpu} \
  --itr 1 --batch_size 256 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done

for batch_size in 16 32 64 128 256
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
  --n_clusters 2 \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 512 \
  --d_state 16 \
  --d_ff 512 \
  --is_flip 1 \
  --dropout 0.1\
  --fc_dropout 0.1 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'batch_size''敏感性' \
  --use_wandb True \
  --train_epochs 30 \
  --patience 5 \
  --lradj 'type3'\
  --pct_start 0.2\
  --gpu ${gpu} \
  --itr 1 --batch_size $batch_size --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done

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
  --n_clusters 2 \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model $d_model \
  --d_state 16 \
  --d_ff 512 \
  --is_flip 1 \
  --dropout 0.1\
  --fc_dropout 0.1 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'd_model''敏感性' \
  --use_wandb True \
  --train_epochs 30 \
  --patience 5 \
  --lradj 'type3'\
  --pct_start 0.2\
  --gpu ${gpu} \
  --itr 1 --batch_size 256 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
  --n_clusters 2 \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 512 \
  --d_state 16 \
  --d_ff $d_ff \
  --is_flip 1 \
  --dropout 0.1\
  --fc_dropout 0.1 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'd_ff''敏感性' \
  --use_wandb True \
  --train_epochs 30 \
  --patience 5 \
  --lradj 'type3'\
  --pct_start 0.2\
  --gpu ${gpu} \
  --itr 1 --batch_size 256 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
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
  --n_clusters 2 \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 512 \
  --d_state $d_state \
  --d_ff 512 \
  --is_flip 1 \
  --dropout 0.1\
  --fc_dropout 0.1 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'd_state''敏感性' \
  --use_wandb True \
  --train_epochs 30 \
  --patience 5 \
  --lradj 'type3'\
  --pct_start 0.2\
  --gpu ${gpu} \
  --itr 1 --batch_size 256 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done
