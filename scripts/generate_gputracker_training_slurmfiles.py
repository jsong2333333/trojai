import itertools
import argparse
from gputracker.gputracker import get_logger, DispatchThread
parser = argparse.ArgumentParser(description='')

parser.add_argument("--train", action="store_true",default=False, help="whether to generate ww scripts for trained models")
parser.add_argument("--prune", action="store_true",default=False, help="whether to generate ww scripts for pruned models")
parser.add_argument("--finetune", action="store_true", default=False, help="whether to generate ww scripts for finetuned models")
parser.add_argument("--base_folder",  default='data', help="put your slurm user name here")
parser.add_argument("--username", default='yefan0726', help="put your slurm user name here")
parser.add_argument("--version",  type=int, default=1, help="version of code")
parser.add_argument("--gpus", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7], help="ids of gpu to use")

args = parser.parse_args()
logger = get_logger('log', 'schedule_subspace.log')
TARGET_NODES = ['steropes', 'steropes', 'steropes', 'steropes']

DATASET='cifar10'
FULL_MODEL_SEEDS=[1, 2, 3]

ARCHS=['preresnet'] #preresnet32
DEPTHS=[230]   #, ,32 86 
DATA_SAMPLE_FARC=[1] # 0.05, 0.2, 0.5, 0.7, 0.9
EPOCH_STOP_INTERVAL=10

RANDOM_LABEL=False  
LABEL_CORRUPT_PROB=0.2
NOISE_TYPE='symmetric'
TRAIN_BATCH_SIZE=64
WIDTHS = [64]
RESUME_PATH = None
TOTAL_EPOCHS=160
SCHEDULE=[80, 120]


full_model_grid = list(itertools.product(
                            ARCHS,
                            DEPTHS,
                            FULL_MODEL_SEEDS,
                            DATA_SAMPLE_FARC,
                            WIDTHS
                            ))

                        
job_nums = len(full_model_grid)
job_counter = 0
BASH_COMMAND_LIST = []

for arch, depth, full_model_seed, sample_frac, width in full_model_grid:

    
    #slurm_file = f'train_full_corrprob{LABEL_CORRUPT_PROB}_{NOISE_TYPE}LabelNoise{RANDOM_LABEL}' + \
    #        f'{DATASET}_{arch}{depth}_WIDTH{width}_data_sample_frac{sample_frac}_bs{TRAIN_BATCH_SIZE}_fullseed{full_model_seed}.sh'

    slurm_cmd =  '#!/bin/bash\n'
    slurm_cmd += '#SBATCH -p rise # partition (queue)\n'
    slurm_cmd += f'#SBATCH -D /home/eecs/{args.username}/ww_prune/cv/rethinking-network-pruning\n'
    slurm_cmd += '##SBATCH --exclude=havoc,r4,r16,atlas\n'
    if job_counter in list(range(0, int(job_nums * 0.25))):
        TARGET_NODE = TARGET_NODES[0]
    elif job_counter in range(int(job_nums * 0.25), int(job_nums * 0.5)):
        TARGET_NODE = TARGET_NODES[1]
    elif job_counter in range(int(job_nums * 0.5), int(job_nums * 0.75)):
        TARGET_NODE = TARGET_NODES[2]
    else:
        TARGET_NODE = TARGET_NODES[3]
    # slurm_cmd += f'#SBATCH --nodelist={TARGET_NODE}\n'

    # slurm_cmd += '#SBATCH -n 1 # number of tasks (i.e. processes)\n'
    # slurm_cmd += '#SBATCH --cpus-per-task=6 # number of cores per task\n'
    # slurm_cmd += '#SBATCH --gres=gpu:1\n'
    # slurm_cmd += '#SBATCH -t 0-12:00 # time requested (D-HH:MM)\n'
    # slurm_cmd += 'pwd\n'
    # slurm_cmd += 'hostname\n'
    # slurm_cmd += 'date\n'
    # slurm_cmd += 'echo starting job...\n'
    # slurm_cmd += 'source ~/.bashrc\n'
    # slurm_cmd += 'conda activate ww_prune\n' 
    # #slurm_cmd += 'pip install torchvision==0.11.2\n' 
    # #slurm_cmd += 'pip install progress\n'
    # slurm_cmd += 'export PYTHONUNBUFFERED=1\n'
    # slurm_cmd += 'export OMP_NUM_THREADS=1\n'                               # limit the trends used 
    # slurm_cmd += f'cd /home/eecs/{args.username}/ww_prune/cv/rethinking-network-pruning\n'
    # if RANDOM_LABEL:
    #     slurm_cmd += f'mkdir -p /{args.base_folder}/{args.username}/data/cv/{DATASET}/random_labels/\n'
    #     slurm_cmd += f'scp -r yefan0726@watson.millennium.berkeley.edu:/{args.base_folder}/{args.username}/{args.base_folder}/cv/{DATASET}/random_labels/{NOISE_TYPE}_label_prob{LABEL_CORRUPT_PROB}_seed{full_model_seed}.pkl /{args.base_folder}/{args.username}/data/cv/{DATASET}/random_labels\n'
    
    # num_samples = int(50000 * sample_frac)
    # slurm_cmd += f'scp -r yefan0726@watson.millennium.berkeley.edu:/{args.base_folder}/{args.username}/data/cv/{DATASET}/subset_indices_{num_samples}_seed1.npy /{args.base_folder}/{args.username}/data/cv/{DATASET}/\n'
    
    # jobs_path = open(f'jobs/scripts/{slurm_file}', 'w+')
    # jobs_path.write('{:s} \n'.format(slurm_cmd))

    cmd =''
    cmd += 'python cifar/weight-level/cifar.py '
    cmd += f'--dataset {DATASET} '
    cmd += f'--arch {arch} '
    cmd += f'--depth {depth} '
    cmd += f'--manualSeed {full_model_seed} '
    if LABEL_CORRUPT_PROB and RANDOM_LABEL:
        cmd += f'--save_dir /{args.base_folder}/yefan0726/checkpoints/cv/{DATASET}/pretrain/{arch}{depth}_w{width}_seed{full_model_seed}_frac{sample_frac}_{NOISE_TYPE}_corrprob{LABEL_CORRUPT_PROB}_randomLabel{RANDOM_LABEL}_epochs{TOTAL_EPOCHS}_sche{SCHEDULE[0]}{SCHEDULE[1]} '
    else:
        cmd += f'--save_dir /{args.base_folder}/yefan0726/checkpoints/cv/{DATASET}/pretrain/{arch}{depth}_seed{full_model_seed}_frac{sample_frac} '
    
    cmd += f'--data-path /{args.base_folder}/{args.username}/data/cv/{DATASET} '
    cmd += f'--data-subsample-frac {sample_frac} '
    cmd += f'--random-labels ' if RANDOM_LABEL else ''
    cmd += f'--random-label-path /{args.base_folder}/{args.username}/data/cv/{DATASET}/random_labels/{NOISE_TYPE}_label_prob{LABEL_CORRUPT_PROB}_seed{full_model_seed}.pkl '
    if LABEL_CORRUPT_PROB:
        cmd += f'--label-corrupt-prob {LABEL_CORRUPT_PROB} '
        cmd += f'--noise-type {NOISE_TYPE} '
    cmd += f'--width {width} '
    cmd += f'--epochs {TOTAL_EPOCHS} '
    cmd += f'--schedule {SCHEDULE[0]} {SCHEDULE[1]} '
    cmd += f'--resume {RESUME_PATH} ' if RESUME_PATH else ''
    cmd += f'--train-batch {TRAIN_BATCH_SIZE} '
    cmd += f'--epoch-stop-intervel {EPOCH_STOP_INTERVAL}\n'       

    # cmd += 'date\n'
    # cmd += f'echo "{arch} {depth} done"\n'    
    # if args.base_folder != 'work': 
    #     if LABEL_CORRUPT_PROB and RANDOM_LABEL: 
    #         cmd += f'scp -r /{args.base_folder}/{args.username}/checkpoints/cv/{DATASET}/pretrain/{arch}{depth}_w{width}_seed{full_model_seed}_frac{sample_frac}_{NOISE_TYPE}_corrprob{LABEL_CORRUPT_PROB}_randomLabel{RANDOM_LABEL}_epochs{TOTAL_EPOCHS}_sche{SCHEDULE[0]}{SCHEDULE[1]} {args.username}@watson.millennium.berkeley.edu:/{args.base_folder}/{args.username}/checkpoints/cv/{DATASET}/pretrain\n'                   
    #     else:
    #         cmd += f'scp -r /{args.base_folder}/{args.username}/checkpoints/cv/{DATASET}/pretrain/{arch}{depth}_seed{full_model_seed}_frac{sample_frac} {args.username}@watson.millennium.berkeley.edu:/{args.base_folder}/{args.username}/checkpoints/cv/{DATASET}/pretrain\n'                   
    
    BASH_COMMAND_LIST.append(cmd)
    #jobs_path.write('{:s} \n'.format(cmd))
 
    # cmd = f'echo "All done"\n'
    # jobs_path.write('{:s} \n'.format(cmd))                 
    # jobs_path.close()
    # job_counter += 1
    # submit_path = open('jobs/slurm_train_submit.sh', 'a+')
    # slurm_output_log = slurm_file.replace('sh', 'log')
    # submit_path.write('#{:s}\n'.format(TARGET_NODE))
    # submit_path.write('{:s}\n'.format(f"sbatch -o /home/eecs/{args.username}/ww_prune/cv/rethinking-network-pruning/jobs/slurm_output/{slurm_output_log} /home/eecs/{args.username}/ww_prune/cv/rethinking-network-pruning/jobs/scripts/{slurm_file}"))

dispatch_thread = DispatchThread("synthetic dataset training", 
                 BASH_COMMAND_LIST, logger, gpu_m_th=500, gpu_list=args.gpus, maxcheck=0)


# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")