# stlt: small training, large test
CUDA_ID=0
TASK_NAME=neuralgw
log_folder=logs_indomain_small_training_large_test
mkdir -p $log_folder

python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name MUTAG --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ngammas9_seed5_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name BZR --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_BZR_ngammas9_seed5_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name COX2 --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_COX2_ngammas9_seed5_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name DHFR --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_DHFR_ngammas9_seed5_parallel.txt

python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name DHFR --n_gammas 9 --seed 9 --model_type parallel | tee $log_folder/log_${TASK_NAME}_DHFR_ngammas9_seed9_parallel.txt


echo "All tasks completed."