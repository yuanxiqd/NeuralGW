log_folder=logs_cdomain_mtest
mkdir -p $log_folder
CUDA_ID=3
TASK_NAME=neuralgw

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name SBM --n_gammas 9 --seed 0 --model_type parallel | tee logs/log_neuralgw_SBM_ngammas9_seed0_parallel.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name REDDIT-BINARY --n_gammas 9 --seed 0 --model_type parallel | tee logs/log_neuralgw_REDDIT-BINARY_ngammas9_seed0_parallel.txt
# CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 main_ddp.py --task neuralgw --ds_name1 COLLAB --ds_name2 REDDIT-BINARY --n_gammas 9 --seed 0 --model_type parallel | tee logs/log_neuralgw_REDDIT-BINARY_ngammas9_seed0_parallel.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 MUTAG --n_gammas 9 --seed 0 --model_type parallel | tee logs/log_neuralgw_mmd0_BZR_PTC_FM_MUTAG_ngammas9_seed0_parallel.txt
# python main.py --task neuralgw --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 MUTAG --n_gammas 9 --seed 0 --model_type parallel | tee logs/log_neuralgw_MUTAG_DD_COLLAB_MUTAG_ngammas9_seed0_parallel.txt
python main_cdomain_mtest.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 DHFR --ds_name2 COX2 --ds_name3 MUTAG --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_DHFR_COX2_MUTAG_ngammas9_seed0_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 MUTAG --n_gammas 9 --seed 1 --model_type parallel | tee $log_folder/log_neuralgw_BZR_PTC_FM_MUTAG_ngammas9_seed1_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 MUTAG --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_neuralgw_BZR_PTC_FM_MUTAG_ngammas9_seed2_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 MUTAG --n_gammas 9 --seed 3 --model_type parallel | tee $log_folder/log_neuralgw_BZR_PTC_FM_MUTAG_ngammas9_seed3_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 MUTAG --n_gammas 9 --seed 4 --model_type parallel | tee $log_folder/log_neuralgw_BZR_PTC_FM_MUTAG_ngammas9_seed4_parallel.txt

# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 DHFR --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_neuralgw_BZR_PTC_FM_DHFR_ngammas9_seed0_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 DHFR --n_gammas 9 --seed 1 --model_type parallel | tee $log_folder/log_neuralgw_BZR_PTC_FM_DHFR_ngammas9_seed1_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 DHFR --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_neuralgw_BZR_PTC_FM_DHFR_ngammas9_seed2_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 DHFR --n_gammas 9 --seed 3 --model_type parallel | tee $log_folder/log_neuralgw_BZR_PTC_FM_DHFR_ngammas9_seed3_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 DHFR --n_gammas 9 --seed 4 --model_type parallel | tee $log_folder/log_neuralgw_BZR_PTC_FM_DHFR_ngammas9_seed4_parallel.txt

# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 COX2 --ds_name3 DHFR --n_gammas 9 --seed 0 --model_type parallel | tee $log_folder/log_neuralgw_BZR_COX2_DHFR_ngammas9_seed0_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 COX2 --ds_name3 DHFR --n_gammas 9 --seed 1 --model_type parallel | tee $log_folder/log_neuralgw_BZR_COX2_DHFR_ngammas9_seed1_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 COX2 --ds_name3 DHFR --n_gammas 9 --seed 2 --model_type parallel | tee $log_folder/log_neuralgw_BZR_COX2_DHFR_ngammas9_seed2_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 COX2 --ds_name3 DHFR --n_gammas 9 --seed 3 --model_type parallel | tee $log_folder/log_neuralgw_BZR_COX2_DHFR_ngammas9_seed3_parallel.txt
# python main_cdomain.py --task neuralgw --ds_name1 BZR --ds_name2 COX2 --ds_name3 DHFR --n_gammas 9 --seed 4 --model_type parallel | tee $log_folder/log_neuralgw_BZR_COX2_DHFR_ngammas9_seed4_parallel.txt

# for seed in {0..4}
# do
#     python main.py --task neuralgw --ds_name MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_202512221523_neuralgw_MUTAG_ngammas9_seed${seed}_parallel.txt
# done

# for seed in {0..0}
# do
#     # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 COLLAB --ds_name2 REDDIT-BINARY --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_REDDIT-BINARY_ngammas9_seed${seed}_parallel.txt
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 MUTAG --ds_name2 BZR --ds_name3 COX2 --ds_name4 PTC_FM --ds_name5 ENZYMES --ds_name6 COLLAB --ds_name7 REDDIT-BINARY  --n_gammas 9 --seed $seed --model_type parallel | tee logs_202512291139/log_neuralgw_MUTAG_BZR_COX2_PTC_FM_ENZYMES_COLLAB_REDDIT-BINARY_ngammas9_seed${seed}_parallel.txt
#     # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 MUTAG --ds_name2 BZR --ds_name3 COX2 --ds_name4 PTC_FM --ds_name5 ENZYMES --ds_name6 COLLAB --ds_name7 IMDB-MULTI --n_gammas 9 --seed $seed --model_type parallel | tee logs_202512291139/log_neuralgw_MUTAG_BZR_COX2_PTC_FM_ENZYMES_COLLAB_IMDB-MULTI_ngammas9_seed${seed}_parallel.txt
# done

# for seed in {0..0}
# do
#     # python main.py --task neuralgw --ds_name MUTAG  --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_mmd0_MUTAG_ngammas9_seed${seed}_parallel.txt
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_mmd1_BZR_PTC_FM_MUTAG_ngammas9_seed${seed}_parallel.txt
# done

# echo "All tasks of BZR + PTC_FM on MUTAG completed."

# for seed in {0..0}
# do
#     # python main.py --task neuralgw --ds_name BZR  --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_mmd0_BZR_ngammas9_seed${seed}_parallel.txt
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 DHFR --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_mmd1_BZR_PTC_FM_DHFR_ngammas9_seed${seed}_parallel.txt
# done

# echo "All tasks of BZR + PTC_FM on DHFR completed."

# for seed in {0..4}
# do
#     # python main.py --task neuralgw --ds_name COX2  --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_mmd0_COX2_ngammas9_seed${seed}_parallel.txt
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 BZR --ds_name2 COX2 --ds_name3 DHFR --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_mmd1_BZR_COX2_DHFR_ngammas9_seed${seed}_parallel.txt
# done

# echo "All tasks of BZR + COX2 on DHFR completed."

# for seed in {0..4}
# do
#     python main.py --task neuralgw --ds_name DHFR  --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_mmd0_DHFR_ngammas9_seed${seed}_parallel.txt
# done

# echo "All tasks on DHFR completed."

echo "All tasks completed."