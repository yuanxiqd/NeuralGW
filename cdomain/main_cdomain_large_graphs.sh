log_folder=logs_cdomain_large_graphs
mkdir -p $log_folder
CUDA_ID=1
TASK_NAME=neuralgw

# python main.py --task neuralgw --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 MUTAG --n_gammas 9 --seed 0 --model_type parallel | tee logs/log_neuralgw_MUTAG_DD_COLLAB_MUTAG_ngammas9_seed0_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 IMDB-MULTI --ds_name2 REDDIT-BINARY --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_IMDB-MULTI_REDDIT-BINARY_ngammas9_seed0_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 COLLAB --ds_name2 REDDIT-BINARY --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_COLLAB_REDDIT-BINARY_ngammas9_seed0_parallel_test.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 COLLAB --ds_name2 REDDIT-BINARY --n_gammas 9 --seed 1 --model_type parallel | tee $log_folder/log_${TASK_NAME}_COLLAB_REDDIT-BINARY_ngammas9_seed1_parallel_test.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 COLLAB --ds_name2 REDDIT-BINARY --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_COLLAB_REDDIT-BINARY_ngammas9_seed2_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 COLLAB --ds_name2 REDDIT-BINARY --n_gammas 9 --seed 3 --model_type parallel | tee $log_folder/log_${TASK_NAME}_COLLAB_REDDIT-BINARY_ngammas9_seed3_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 COLLAB --ds_name2 REDDIT-BINARY --n_gammas 9 --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_COLLAB_REDDIT-BINARY_ngammas9_seed4_parallel.txt

# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 REDDIT-BINARY --ds_name2 COLLAB --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_REDDIT-BINARY_COLLAB_ngammas9_seed0_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 REDDIT-BINARY --ds_name2 COLLAB --n_gammas 9 --seed 1 --model_type parallel | tee $log_folder/log_${TASK_NAME}_REDDIT-BINARY_COLLAB_ngammas9_seed1_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 REDDIT-BINARY --ds_name2 COLLAB --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_REDDIT-BINARY_COLLAB_ngammas9_seed2_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 REDDIT-BINARY --ds_name2 COLLAB --n_gammas 9 --seed 3 --model_type parallel | tee $log_folder/log_${TASK_NAME}_REDDIT-BINARY_COLLAB_ngammas9_seed3_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 REDDIT-BINARY --ds_name2 COLLAB --n_gammas 9 --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_REDDIT-BINARY_COLLAB_ngammas9_seed4_parallel.txt

echo "All tasks completed."