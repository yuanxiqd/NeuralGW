log_folder=logs_cdomain_multiple2single_ok
mkdir -p $log_folder
CUDA_ID=1
TASK_NAME=neuralgw

# EMBEDDING_DIM_FOR_MMD_LAYER=64
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 IMDB-MULTI --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_IMDB-MULTI_ngammas9_seed0_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 IMDB-MULTI --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 1 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_IMDB-MULTI_ngammas9_seed1_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 IMDB-MULTI --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_IMDB-MULTI_ngammas9_seed2_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 IMDB-MULTI --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 3 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_IMDB-MULTI_ngammas9_seed3_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 IMDB-MULTI --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_IMDB-MULTI_ngammas9_seed4_parallel.txt

# EMBEDDING_DIM_FOR_MMD_LAYER=64
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 ENZYMES --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_ENZYMES_ngammas9_seed0_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 ENZYMES --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 1 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_ENZYMES_ngammas9_seed1_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 ENZYMES --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_ENZYMES_ngammas9_seed2_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 ENZYMES --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 3 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_ENZYMES_ngammas9_seed3_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 ENZYMES --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_ENZYMES_ngammas9_seed4_parallel.txt

EMBEDDING_DIM_FOR_MMD_LAYER=128
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 DHFR --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_DHFR_ngammas9_seed0_parallel.txt
# python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 DHFR --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 1 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_DHFR_ngammas9_seed1_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 DHFR --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_DHFR_ngammas9_seed2_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 DHFR --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 3 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_DHFR_ngammas9_seed3_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DD --ds_name3 COLLAB --ds_name4 DHFR --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DD_COLLAB_DHFR_ngammas9_seed4_parallel.txt

EMBEDDING_DIM_FOR_MMD_LAYER=64
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 IMDB-MULTI --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_IMDB-MULTI_ngammas9_seed0_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 IMDB-MULTI --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 1 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_IMDB-MULTI_ngammas9_seed1_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 IMDB-MULTI --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_IMDB-MULTI_ngammas9_seed2_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 IMDB-MULTI --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 3 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_IMDB-MULTI_ngammas9_seed3_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 IMDB-MULTI --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_IMDB-MULTI_ngammas9_seed4_parallel.txt

EMBEDDING_DIM_FOR_MMD_LAYER=64
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 ENZYMES --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ENZYMES_ngammas9_seed0_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 ENZYMES --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 1 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ENZYMES_ngammas9_seed1_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 ENZYMES --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ENZYMES_ngammas9_seed2_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 ENZYMES --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 3 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ENZYMES_ngammas9_seed3_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 ENZYMES --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_ENZYMES_ngammas9_seed4_parallel.txt

EMBEDDING_DIM_FOR_MMD_LAYER=64
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DHFR --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 0 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DHFR_ngammas9_seed0_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DHFR --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 1 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DHFR_ngammas9_seed1_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DHFR --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 2 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DHFR_ngammas9_seed2_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DHFR --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 3 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DHFR_ngammas9_seed3_parallel.txt
python main_cdomain.py --cuda_id $CUDA_ID --task $TASK_NAME --ds_name1 MUTAG --ds_name2 DHFR --n_gammas 9 --embedding_dim_for_mmd_layer $EMBEDDING_DIM_FOR_MMD_LAYER --seed 4 --model_type parallel | tee $log_folder/log_${TASK_NAME}_MUTAG_DHFR_ngammas9_seed4_parallel.txt

echo "All tasks completed."