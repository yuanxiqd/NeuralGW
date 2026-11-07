# stlt: small training, large test
CUDA_ID=0
TASK_NAME=neuralgw

TEST_RATIO=0.9
log_folder=logs_indomain_small_training_large_test_test_ratio_${TEST_RATIO}
mkdir -p $log_folder
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name MUTAG --test_ratio $TEST_RATIO --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ngammas9_seed0_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name BZR --test_ratio $TEST_RATIO --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_BZR_ngammas9_seed0_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name COX2 --test_ratio $TEST_RATIO --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_COX2_ngammas9_seed0_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name DHFR --test_ratio $TEST_RATIO --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_DHFR_ngammas9_seed0_parallel.txt

python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name DHFR --test_ratio $TEST_RATIO --n_gammas 9 --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_DHFR_ngammas9_seed4_parallel.txt

python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name MUTAG --test_ratio $TEST_RATIO --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ngammas9_seed5_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name BZR --test_ratio $TEST_RATIO --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_BZR_ngammas9_seed5_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name COX2 --test_ratio $TEST_RATIO --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_COX2_ngammas9_seed5_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name DHFR --test_ratio $TEST_RATIO --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_DHFR_ngammas9_seed5_parallel.txt

python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name DHFR --test_ratio $TEST_RATIO --n_gammas 9 --seed 9 --model_type parallel | tee $log_folder/log_${TASK_NAME}_DHFR_ngammas9_seed9_parallel.txt

TEST_RATIO=0.4
log_folder=logs_indomain_small_training_large_test_test_ratio_${TEST_RATIO}
mkdir -p $log_folder
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name MUTAG --test_ratio $TEST_RATIO --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ngammas9_seed0_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name BZR --test_ratio $TEST_RATIO --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_BZR_ngammas9_seed0_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name COX2 --test_ratio $TEST_RATIO --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_COX2_ngammas9_seed0_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name DHFR --test_ratio $TEST_RATIO --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_DHFR_ngammas9_seed0_parallel.txt

python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name DHFR --test_ratio $TEST_RATIO --n_gammas 9 --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_DHFR_ngammas9_seed4_parallel.txt

python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name MUTAG --test_ratio $TEST_RATIO --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ngammas9_seed5_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name BZR --test_ratio $TEST_RATIO --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_BZR_ngammas9_seed5_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name COX2 --test_ratio $TEST_RATIO --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_COX2_ngammas9_seed5_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name DHFR --test_ratio $TEST_RATIO --n_gammas 9 --seed 5 --model_type parallel | tee $log_folder/log_${TASK_NAME}_DHFR_ngammas9_seed5_parallel.txt

python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name DHFR --test_ratio $TEST_RATIO --n_gammas 9 --seed 9 --model_type parallel | tee $log_folder/log_${TASK_NAME}_DHFR_ngammas9_seed9_parallel.txt

echo "All tasks completed."