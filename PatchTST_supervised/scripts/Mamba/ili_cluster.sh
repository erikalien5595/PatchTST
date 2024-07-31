export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=2

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=Mamba

root_path_name=./dataset/illness/
data_path_name=national_illness.csv
model_id_name=ILI # 如果是聚类后的模型，model_id_name后面再加上_cluster
data_name=custom

random_seed=2024
n_clusters=3
for pred_len in 24 36 48 60
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
      --e_layers 2 \
      --n_heads 16 \
      --d_model 512 \
      --d_state 16 \
      --d_ff 512 \
      --is_flip 1 \
      --dropout 0.2\
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Cluster'$n_clusters'Flip' \
      --train_epochs 10 \
      --patience 5\
      --lradj '5'\
      --pct_start 0.2\
      --gpu ${gpu} \
      --itr 1 --batch_size 16 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done

exit 0

for pred_len in 24 36 48 60
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
      --e_layers 2 \
      --n_heads 16 \
      --d_model 512 \
      --d_state 16 \
      --d_ff 512 \
      --is_flip 1 \
      --dropout 0.3\
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Cluster2Flip' \
      --train_epochs 20 \
      --patience 10\
      --lradj '5'\
      --pct_start 0.2\
      --gpu ${gpu} \
      --itr 1 --batch_size 32 --learning_rate 0.0005 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
# 24 48 lr=0.0005, 36 lr=0.0001
exit 0