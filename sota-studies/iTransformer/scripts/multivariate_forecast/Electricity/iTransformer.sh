export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path /kuacc/users/hgorgulu22/hpc_run/workfolder/Projects/AlarmPrediction/datasets/iTransformer_datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data process \ #change this dataset name
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path /kuacc/users/hgorgulu22/hpc_run/workfolder/Projects/AlarmPrediction/datasets/iTransformer_datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path /kuacc/users/hgorgulu22/hpc_run/workfolder/Projects/AlarmPrediction/datasets/iTransformer_datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path /kuacc/users/hgorgulu22/hpc_run/workfolder/Projects/AlarmPrediction/datasets/iTransformer_datasets/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1