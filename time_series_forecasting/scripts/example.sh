#!/usr/bin/env bash
set -euo pipefail

# Run from the forecasting module root, regardless of where this script is called.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p ./logs/LongForecasting


seq_len=336
model_name=DLinear


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_none_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 >logs/LongForecasting/$model_name'_'ETTh2_none_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fmask_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_mask \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_mask_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fmix_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_mix \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_mix_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fpool_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_pool \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_pool_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fadd_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_add \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_add_$seq_len'_'96.log
  



python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_robustp_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method robust_p \
  --aug_rate 0.5 \
  --K_num 1\
  --seg_ratio 0.8 >logs/LongForecasting/$model_name'_'ETTh2_robustp_$seq_len'_'96.log
  
  
python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_robustm_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method robust_m \
  --aug_rate 0.1 \
  --K_num 3\
  --seg_ratio 0.2 >logs/LongForecasting/$model_name'_'ETTh2_robustm_$seq_len'_'96.log


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_upsamplev1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method upsample \
  --aug_rate 0.1 >logs/LongForecasting/$model_name'_'ETTh2_upsamplev1_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_upsamplev2_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method upsample \
  --aug_rate 0.4 >logs/LongForecasting/$model_name'_'ETTh2_upsamplev2_$seq_len'_'96.log


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_wmask_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method w_mask \
  --rates "[0.5, 0.3, 0.9, 0.9, 0.0, 0.0, 0.0]" \
  --wavelet 'db2' \
  --level 3 \
  --sampling_rate 0.2   >logs/LongForecasting/$model_name'_'ETTh2_wmask_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_wmix_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method w_mix \
  --rates "[0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]" \
  --wavelet 'db3' \
  --level 1 \
  --sampling_rate 0.2  >logs/LongForecasting/$model_name'_'ETTh2_wmix_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 2  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev1_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev2_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 3  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev2_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev3_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 4  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev3_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev4_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 8 >logs/LongForecasting/$model_name'_'ETTh2_domshufflev4_$seq_len'_'96.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_tps_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method tps \
  --aug_rate 1.0 \
  --aug_patch_len 32 \
  --aug_stride 5  >logs/LongForecasting/$model_name'_'ETTh2_tps_$seq_len'_'96.log

  
python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_mbb_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method mbb \
  --block_size 96  >logs/LongForecasting/$model_name'_'ETTh2_mbb_$seq_len'_'96.log 
  
  
python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_asd_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method asd \
  --aug_rate 0.5  >logs/LongForecasting/$model_name'_'ETTh2_asd_$seq_len'_'96.log 
  

echo "DLinear ETTh2 pred 96 finsihed"


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_none_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 >logs/LongForecasting/$model_name'_'ETTh2_none_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fmask_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_mask \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_mask_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fmix_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_mix \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_mix_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fpool_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_pool \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_pool_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fadd_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_add \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_add_$seq_len'_'192.log
  


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_robustp_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method robust_p \
  --aug_rate 0.5 \
  --K_num 1\
  --seg_ratio 0.8 >logs/LongForecasting/$model_name'_'ETTh2_robustp_$seq_len'_'192.log


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_robustm_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method robust_m \
  --aug_rate 0.1 \
  --K_num 3\
  --seg_ratio 0.2 >logs/LongForecasting/$model_name'_'ETTh2_robustm_$seq_len'_'192.log
  
  
python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_upsamplev1_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method upsample \
  --aug_rate 0.1 >logs/LongForecasting/$model_name'_'ETTh2_upsamplev1_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_upsamplev2_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method upsample \
  --aug_rate 0.4 >logs/LongForecasting/$model_name'_'ETTh2_upsamplev2_$seq_len'_'192.log


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_wmask_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method w_mask \
  --rates "[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]" \
  --wavelet 'db3' \
  --level 1 \
  --sampling_rate 0.2 >logs/LongForecasting/$model_name'_'ETTh2_wmask_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_wmix_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method w_mix \
  --rates "[1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]" \
  --wavelet 'db3' \
  --level 1 \
  --sampling_rate 0.8   >logs/LongForecasting/$model_name'_'ETTh2_wmix_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev1_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 2  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev1_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev2_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 3  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev2_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev3_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 4  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev3_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev4_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 8 >logs/LongForecasting/$model_name'_'ETTh2_domshufflev4_$seq_len'_'192.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_tps_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method tps \
  --aug_rate 1.0 \
  --aug_patch_len 32 \
  --aug_stride 5  >logs/LongForecasting/$model_name'_'ETTh2_tps_$seq_len'_'192.log

  
python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_mbb_$seq_len'_'96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method mbb \
  --block_size 96  >logs/LongForecasting/$model_name'_'ETTh2_mbb_$seq_len'_'192.log 
  
  
python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_asd_$seq_len'_'192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method asd \
  --aug_rate 0.5  >logs/LongForecasting/$model_name'_'ETTh2_asd_$seq_len'_'192.log 
  

echo "DLinear ETTh2 pred 192 finsihed"


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_none_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 >logs/LongForecasting/$model_name'_'ETTh2_none_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fmask_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_mask \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_mask_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fmix_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_mix \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_mix_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fpool_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_pool \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_pool_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fadd_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_add \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_add_$seq_len'_'336.log
  


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_robustp_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method robust_p \
  --aug_rate 0.5 \
  --K_num 1\
  --seg_ratio 0.8 >logs/LongForecasting/$model_name'_'ETTh2_robustp_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_robustm_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method robust_m \
  --aug_rate 0.1 \
  --K_num 3\
  --seg_ratio 0.2 >logs/LongForecasting/$model_name'_'ETTh2_robustm_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_upsamplev1_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method upsample \
  --aug_rate 0.1 >logs/LongForecasting/$model_name'_'ETTh2_upsamplev1_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_upsamplev2_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method upsample \
  --aug_rate 0.4 >logs/LongForecasting/$model_name'_'ETTh2_upsamplev2_$seq_len'_'336.log


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_wmask_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method w_mask \
  --rates "[0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]" \
  --wavelet 'db25' \
  --level 1 \
  --sampling_rate 0.8  >logs/LongForecasting/$model_name'_'ETTh2_wmask_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_wmix_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method w_mix \
  --rates "[0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]" \
  --wavelet 'db3' \
  --level 1 \
  --sampling_rate 0.8   >logs/LongForecasting/$model_name'_'ETTh2_wmix_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev1_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 2  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev1_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev2_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 3  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev2_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev3_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 4  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev3_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev4_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 8 >logs/LongForecasting/$model_name'_'ETTh2_domshufflev4_$seq_len'_'336.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_tps_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method tps \
  --aug_rate 1.0 \
  --aug_patch_len 32 \
  --aug_stride 5  >logs/LongForecasting/$model_name'_'ETTh2_tps_$seq_len'_'336.log


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_mbb_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method mbb \
  --block_size 96  >logs/LongForecasting/$model_name'_'ETTh2_mbb_$seq_len'_'336.log 
  
  
python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_asd_$seq_len'_'336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method asd \
  --aug_rate 0.5  >logs/LongForecasting/$model_name'_'ETTh2_asd_$seq_len'_'336.log 


echo "DLinear ETTh2 pred 336 finsihed"


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_none_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 >logs/LongForecasting/$model_name'_'ETTh2_none_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fmask_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_mask \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_mask_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fmix_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_mix \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_mix_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fpool_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_pool \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_pool_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_fadd_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method f_add \
  --aug_rate 0.5 >logs/LongForecasting/$model_name'_'ETTh2_f_add_$seq_len'_'720.log
  


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_robustp_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method robust_p \
  --aug_rate 0.5 \
  --K_num 1\
  --seg_ratio 0.8 >logs/LongForecasting/$model_name'_'ETTh2_robustp_$seq_len'_'720.log
  
  

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_robustm_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method robust_m \
  --aug_rate 0.1 \
  --K_num 3\
  --seg_ratio 0.2 >logs/LongForecasting/$model_name'_'ETTh2_robustm_$seq_len'_'720.log
  
  
python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_upsamplev1_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method upsample \
  --aug_rate 0.1 >logs/LongForecasting/$model_name'_'ETTh2_upsamplev1_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_upsamplev2_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method upsample \
  --aug_rate 0.4 >logs/LongForecasting/$model_name'_'ETTh2_upsamplev2_$seq_len'_'720.log


python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_wmask_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method w_mask \
  --rates "[0.4, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]" \
  --wavelet 'db1' \
  --level 1 \
  --sampling_rate 0.2  >logs/LongForecasting/$model_name'_'ETTh2_wmask_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_wmix_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method w_mix \
  --rates "[0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]" \
  --wavelet 'db5' \
  --level 1 \
  --sampling_rate 0.8   >logs/LongForecasting/$model_name'_'ETTh2_wmix_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev1_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 2  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev1_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev2_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 3  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev2_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev3_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 4  >logs/LongForecasting/$model_name'_'ETTh2_domshufflev3_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_domshufflev4_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method dom_shuffle \
  --aug_rate 8 >logs/LongForecasting/$model_name'_'ETTh2_domshufflev4_$seq_len'_'720.log

python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_tps_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method tps \
  --aug_rate 1.0 \
  --aug_patch_len 32 \
  --aug_stride 5  >logs/LongForecasting/$model_name'_'ETTh2_tps_$seq_len'_'720.log


  
python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_mbb_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method mbb \
  --block_size 96  >logs/LongForecasting/$model_name'_'ETTh2_mbb_$seq_len'_'720.log 
  
  
python -u run_longExp.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_asd_$seq_len'_'720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp-5itr-all-DLINEAR' \
  --itr 5 \
  --batch_size 32 \
  --learning_rate 0.05 \
  --in_batch_augmentation \
  --aug_method asd \
  --aug_rate 0.5  >logs/LongForecasting/$model_name'_'ETTh2_asd_$seq_len'_'720.log 
  
  
echo "DLinear ETTh2 pred 720 finsihed"


