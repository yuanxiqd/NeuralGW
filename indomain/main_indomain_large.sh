# stlt: small training, large test
CUDA_ID=0
TASK_NAME=neuralgw

TEST_RATIO=0.2
log_folder=logs_indomain_stlt_ratio_${TEST_RATIO}_test
mkdir -p $log_folder
python main_indomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name REDDIT-BINARY --test_ratio $TEST_RATIO --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_REDDIT-BINARY_ngammas9_seed0_parallel.txt

echo "All tasks completed."