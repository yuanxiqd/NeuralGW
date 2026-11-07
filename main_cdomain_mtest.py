import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import random
import numpy as np
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.datasets import TUDataset
from utils import funcs
import json
from sklearn.model_selection import train_test_split
import timeit
from copy import deepcopy

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # For single GPU
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False # Disable auto-tuner for conv layers
    torch.use_deterministic_algorithms(True) # Ensure deterministic algorithms

def seed_worker(worker_id: int) -> None:
    worker_id = torch.initial_seed() % 2**32
    np.random.seed(worker_id)
    random.seed(worker_id)

if __name__ == "__main__":
    ################################################
    # 0. argparse
    ################################################
    parser = argparse.ArgumentParser(description='NeuralGW')
    parser.add_argument('--task', type=str, default='neuralgw', help='Task to perform (default: neuralgw)')
    parser.add_argument('--cuda_id', type=int, default=0, help='CUDA device ID (default: 0)')
    parser.add_argument('--switch', type=str, default='gin', help='Module to use (default: gin)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('--n_gammas', type=int, default=9, help='Number of gamma values (default: 9)')
    parser.add_argument('--model_type', type=str, default='parallel', help='Type of model: serial or parallel (default: parallel)')
    # parser.add_argument('--ds_name', type=str, default='MUTAG', help='Dataset name (default: MUTAG)') # used for indomain training
    parser.add_argument('--ds_name1', type=str, default=None, help='Dataset name (default: MUTAG)') # used for crossdomain training
    parser.add_argument('--ds_name2', type=str, default=None, help='Dataset name (default: BZR)') # used for crossdomain training
    parser.add_argument('--ds_name3', type=str, default=None, help='Dataset name (default: COX2)') # used for crossdomain training
    # parser.add_argument('--ds_name4', type=str, default='PTC_FM', help='Dataset name (default: PTC_FM)')
    # parser.add_argument('--ds_name5', type=str, default='DHFR', help='Dataset name (default: DHFR)')
    args = parser.parse_args()

    ################################################
    # 1. DDP initialization
    ################################################
    # dist.init_process_group(backend='nccl')
    # local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)


    ################################################
    # 2. Dataset setup (train/test split)
    ################################################
    # task = 'neuralgw'  # Options: 'neuralgw', 'neuralgw_n', 'neuralgw_tp', 'neuralgw_tp_n'
    task = args.task
    print(f'[Warning] I have set task to {task}. Change it in the code if needed.')

    if task == 'neuralgw':
        config_file_name = 'main.json'
    elif task == 'neuralgw_ablation':
        config_file_name = 'main_ablation.json'
    elif task == 'neuralgw_n':
        config_file_name = 'main_n.json'
    elif task == 'neuralgw_tp':
        config_file_name = 'main_tp.json'
    elif task == 'neuralgw_tp_n':
        config_file_name = 'main_tp_n.json'
    elif task == 'neuralgw_gps':
        config_file_name = 'main_gps.json'
    elif task == 'neuralgw_gps_n':
        config_file_name = 'main_gps_n.json'
    else:
        raise ValueError(f'Unexpected value of task: {task}')

    # Load configuration from JSON file
    with open(f"configs/{config_file_name}", "r") as f:
        config = json.load(f)

    # override config with argparse args
    config['seed'] = args.seed
    config['cuda_id'] = args.cuda_id
    config['model_type'] = args.model_type
    config['switch'] = args.switch
    # config['n_gammas'] = args.n_gammas
    # config['ds_name'] = args.ds_name # used for indomain training

    if hasattr(args, 'ds_name2') is False:
        # for indomain training
        ds_name_seq = [args.ds_name] # used for indomain training
        if ds_name_seq == ['SBM']:
            testset = 'SBM' # used for indomain training of SBM
        else:
            testset = 'global' # used for indomain training
    else:
        # for crossdomain training
        # # ds_name_seq = [args.ds_name1, args.ds_name2]  # used for crossdomain training
        # ds_name_seq = [args.ds_name1, args.ds_name2, args.ds_name3]  # used for crossdomain training
        # # ds_name_seq = [args.ds_name1, args.ds_name2, args.ds_name3, args.ds_name4]  # used for crossdomain training
        # testset = args.ds_name3  # used for crossdomain training

        # iterate to collect all ds_name{i} arguments
        i = 0
        ds_name_seq = []
        while True:
            ds_name_arg = f'ds_name{i+1}'
            if hasattr(args, ds_name_arg) and getattr(args, ds_name_arg) is not None:
                ds_name_value = getattr(args, ds_name_arg)
                ds_name_seq.append(ds_name_value)
                i += 1
            else:
                break
        testset = ds_name_seq[-2:]
    config['n_gammas'] = args.n_gammas

    # if dist.get_rank() == 0:
    # configure the hyperparameters
    print(f'Script name: {config["name"]}')
    set_seed(config["seed"])
    print(f'Seed set to: {config["seed"]}')
    print(100*'-')
    print(f'Configuration: {json.dumps(config, indent=4, sort_keys=True)}')
    print(100*'-')
    print(f'[Warning] task: {task}')
    if task == 'neuralgw_ablation': print(f'[Warning] Switch: {config["switch"]}')
    print(f'[Warning] Dataset sequence: {ds_name_seq}')
    print(f'[Warning] Testset: {testset}')
    print(f'[Warning] Model type: {config["model_type"]}')
    print(f'[Warning] num_gps_layers: {config.get("num_gps_layers", "N/A")}')
    print(f'[Warning] n_gammas: {config["n_gammas"]}')
    print(100*'-')
    device = torch.device(f"cuda:{config['cuda_id']}" if torch.cuda.is_available() else "cpu")
    print(f'[Warning] Using device: {device}')

    ################################################
    # 3. Model + wrap with DDP
    ################################################
    if task == 'neuralgw':
        from models.neuralgw_mtest import NeuralGW
        model = NeuralGW(
            ds_name_seq=ds_name_seq,
            testset=testset,
            is_naive=False,
            config=config,
            device=device
        ).to(device)
    elif task == 'neuralgw_ablation':
        from models.neuralgw_mtest import NeuralGW_Ablation
        model = NeuralGW_Ablation(
            ds_name_seq=ds_name_seq,
            testset=testset,
            is_naive=False,
            config=config,
            device=device
        ).to(device)
    elif task == 'neuralgw_n':
        from models.neuralgw_mtest import NeuralGW
        model = NeuralGW(
            ds_name_seq=ds_name_seq,
            testset=testset,
            is_naive=True,
            config=config,
            device=device
        ).to(device)
    elif task == 'neuralgw_tp':
        from models.neuralgw_mtest import NeuralGW_TP
        model = NeuralGW_TP(
            ds_name_seq=ds_name_seq,
            testset=testset,
            is_naive=False,
            config=config,
            device=device
        ).to(device)
    elif task == 'neuralgw_tp_n':
        from models.neuralgw_mtest import NeuralGW_TP
        model = NeuralGW_TP(
            ds_name_seq=ds_name_seq,
            testset=testset,
            is_naive=True,
            config=config,
            device=device
        ).to(device)
    elif task == 'neuralgw_gps':
        from models.neuralgw_mtest import NeuralGW_GPS
        model = NeuralGW_GPS(
            ds_name_seq=ds_name_seq,
            testset=testset,
            is_naive=False,
            config=config,
            device=device
        ).to(device)
    elif task == 'neuralgw_gps_n':
        from models.neuralgw_mtest import NeuralGW_GPS
        model = NeuralGW_GPS(
            ds_name_seq=ds_name_seq,
            testset=testset,
            is_naive=True,
            config=config,
            device=device
        ).to(device)
    else:
        raise ValueError(f'Unexpected value of task: {task}')

    # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    # if dist.get_rank() == 0:
    print(f'Model: {model}')
    print(100*'-')

    ################################################
    # 4. Training
    ################################################
    # model.module.train_model(seed_worker=seed_worker)
    model.train_model(seed_worker=seed_worker)
    # if dist.get_rank() == 0:
    # model.module.save_distance_matrix_on_testset()
    print(f'Training completed.')
    print(f'n_gammas: {config["n_gammas"]}')
    print(100*'-')
    print(f'seed: {config["seed"]}')
    print(f'model_type: {config["model_type"]}')

    ################################################
    # 5. Postprocessing (only rank 0)
    ################################################
    # if dist.is_initialized():
    #     dist.destroy_process_group()