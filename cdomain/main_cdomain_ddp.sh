log_folder=logs_202601211530_early_stop
mkdir -p $log_folder

# for seed in {0..0}
# do
#     # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name COX2 --num_gin_layers 1 --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_COX2_ngin1_ngammas9_seed${seed}_parallel.txt
#     # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 MUTAG --ds_name2 ENZYMES --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_MUTAG_ENZYMES_ngammas9_seed${seed}_parallel.txt
#     # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_MUTAG_DD_COLLAB_MUTAG_ngammas9_seed${seed}_parallel.txt
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee $log_folder/log_neuralgw_BZR_PTC_FM_MUTAG_ngammas9_seed${seed}_parallel.txt
#     # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_noon_MUTAG_ngammas9_seed${seed}_parallel.txt
#     # python main.py --task neuralgw --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_MUTAG_DD_COLLAB_MUTAG_ngammas9_seed${seed}_parallel.txt
# done

for seed in {0..4}
do
    # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name COX2 --num_gin_layers 1 --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_COX2_ngin1_ngammas9_seed${seed}_parallel.txt
    # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 MUTAG --ds_name2 ENZYMES --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_MUTAG_ENZYMES_ngammas9_seed${seed}_parallel.txt
    # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_MUTAG_DD_COLLAB_MUTAG_ngammas9_seed${seed}_parallel.txt
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 BZR --ds_name2 PTC_FM --ds_name3 DHFR --n_gammas 9 --seed $seed --model_type parallel | tee $log_folder/log_neuralgw_BZR_PTC_FM_DHFR_ngammas9_seed${seed}_parallel.txt
    # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_noon_MUTAG_ngammas9_seed${seed}_parallel.txt
    # python main.py --task neuralgw --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_MUTAG_DD_COLLAB_MUTAG_ngammas9_seed${seed}_parallel.txt
done

# for seed in {4..4}
# do
#     # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name COX2 --num_gin_layers 1 --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_COX2_ngin1_ngammas9_seed${seed}_parallel.txt
#     # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 MUTAG --ds_name2 ENZYMES --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_MUTAG_ENZYMES_ngammas9_seed${seed}_parallel.txt
#     # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_MUTAG_DD_COLLAB_MUTAG_ngammas9_seed${seed}_parallel.txt
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name1 BZR --ds_name2 COX2 --ds_name3 DHFR --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_20251214night_neuralgw_BZR_COX2_DHFR_ngammas9_seed${seed}_parallel.txt
#     # CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_ddp.py --task neuralgw --ds_name MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_noon_MUTAG_ngammas9_seed${seed}_parallel.txt
#     # python main.py --task neuralgw --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 MUTAG --n_gammas 9 --seed $seed --model_type parallel | tee logs/log_neuralgw_MUTAG_DD_COLLAB_MUTAG_ngammas9_seed${seed}_parallel.txt
# done

echo "All tasks completed."