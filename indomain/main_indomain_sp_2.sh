# sp: shortest path
CUDA_ID=2
TASK_NAME=neuralgw
TEST_RATIO=0.2

# DS_NAME=MUTAG
# ################################################
# GW_COST_MODE=adj
# PE_COST_MODE=sp
# PE_MODE=svd
# ################################################
# log_folder=logs_indomain_${GW_COST_MODE}_${PE_COST_MODE}_${PE_MODE}_ratio_${TEST_RATIO}
# mkdir -p $log_folder
# python main_indomain.py --cuda_id $CUDA_ID --pe_cost_mode $PE_COST_MODE --gw_cost_mode $GW_COST_MODE --pe_mode $PE_MODE --task $TASK_NAME --ds_name $DS_NAME --test_ratio $TEST_RATIO --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_${DS_NAME}_ngammas9_seed2_parallel.txt

# DS_NAME=MUTAG_DIR
# ################################################
# GW_COST_MODE=adj
# PE_COST_MODE=sp
# PE_MODE=svd
# ################################################
# log_folder=logs_indomain_${GW_COST_MODE}_${PE_COST_MODE}_${PE_MODE}_ratio_${TEST_RATIO}
# mkdir -p $log_folder
# python main_indomain.py --cuda_id $CUDA_ID --pe_cost_mode $PE_COST_MODE --gw_cost_mode $GW_COST_MODE --pe_mode $PE_MODE --task $TASK_NAME --ds_name $DS_NAME --test_ratio $TEST_RATIO --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_${DS_NAME}_ngammas9_seed2_parallel.txt

# DS_NAME=MUTAG_DIR
# ################################################
# GW_COST_MODE=sp
# PE_COST_MODE=sp
# PE_MODE=svd
# ################################################
# log_folder=logs_indomain_${GW_COST_MODE}_${PE_COST_MODE}_${PE_MODE}_ratio_${TEST_RATIO}_dir
# mkdir -p $log_folder
# python main_indomain.py --cuda_id $CUDA_ID --pe_cost_mode $PE_COST_MODE --gw_cost_mode $GW_COST_MODE --pe_mode $PE_MODE --task $TASK_NAME --ds_name $DS_NAME --test_ratio $TEST_RATIO --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_${DS_NAME}_ngammas9_seed2_parallel.txt

# ################################################
# GW_COST_MODE=sp
# PE_COST_MODE=adj
# PE_MODE=LapPE
# ################################################
# log_folder=logs_indomain_${GW_COST_MODE}_${PE_COST_MODE}_${PE_MODE}_ratio_${TEST_RATIO}_dir
# mkdir -p $log_folder
# python main_indomain.py --cuda_id $CUDA_ID --pe_cost_mode $PE_COST_MODE --gw_cost_mode $GW_COST_MODE --pe_mode $PE_MODE --task $TASK_NAME --ds_name $DS_NAME --test_ratio $TEST_RATIO --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_${DS_NAME}_ngammas9_seed2_parallel.txt

# ################################################
# GW_COST_MODE=sp
# PE_COST_MODE=adj
# PE_MODE=RWSE
# ################################################
# log_folder=logs_indomain_${GW_COST_MODE}_${PE_COST_MODE}_${PE_MODE}_ratio_${TEST_RATIO}_dir
# mkdir -p $log_folder
# python main_indomain.py --cuda_id $CUDA_ID --pe_cost_mode $PE_COST_MODE --gw_cost_mode $GW_COST_MODE --pe_mode $PE_MODE --task $TASK_NAME --ds_name $DS_NAME --test_ratio $TEST_RATIO --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_${DS_NAME}_ngammas9_seed2_parallel.txt

DS_NAME=MUTAG_DIR
################################################
GW_COST_MODE=sp
PE_COST_MODE=adj
PE_MODE=RWSE
################################################
log_folder=logs_indomain_${GW_COST_MODE}_${PE_COST_MODE}_${PE_MODE}_ratio_${TEST_RATIO}_dir
mkdir -p $log_folder
python main_indomain.py --cuda_id $CUDA_ID --pe_cost_mode $PE_COST_MODE --gw_cost_mode $GW_COST_MODE --pe_mode $PE_MODE --task $TASK_NAME --ds_name $DS_NAME --test_ratio $TEST_RATIO --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_${DS_NAME}_ngammas9_seed0_parallel.txt
python main_indomain.py --cuda_id $CUDA_ID --pe_cost_mode $PE_COST_MODE --gw_cost_mode $GW_COST_MODE --pe_mode $PE_MODE --task $TASK_NAME --ds_name $DS_NAME --test_ratio $TEST_RATIO --n_gammas 9 --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_${DS_NAME}_ngammas9_seed4_parallel.txt

echo "All tasks completed."