CUDA_ID=3
TASK_NAME=neuralgw_tp
log_folder=logs_indomain_tp
mkdir -p $log_folder

# python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name MUTAG --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ngammas9_seed0_parallel.txt
# python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name MUTAG --n_gammas 9 --seed 1 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ngammas9_seed1_parallel.txt
# python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name MUTAG --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ngammas9_seed2_parallel.txt
# python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name MUTAG --n_gammas 9 --seed 3 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ngammas9_seed3_parallel.txt
# python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name MUTAG --n_gammas 9 --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ngammas9_seed4_parallel.txt

# python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name BZR --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_BZR_ngammas9_seed0_parallel.txt
# python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name BZR --n_gammas 9 --seed 1 --model_type parallel | tee $log_folder/log_${TASK_NAME}_BZR_ngammas9_seed1_parallel.txt
# python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name BZR --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_BZR_ngammas9_seed2_parallel.txt
# python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name BZR --n_gammas 9 --seed 3 --model_type parallel | tee $log_folder/log_${TASK_NAME}_BZR_ngammas9_seed3_parallel.txt
# python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name BZR --n_gammas 9 --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_BZR_ngammas9_seed4_parallel.txt

echo "All tasks completed."