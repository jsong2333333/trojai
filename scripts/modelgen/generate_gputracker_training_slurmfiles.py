import itertools
import argparse
from gputracker.gputracker import get_logger, DispatchThread
import os
import csv
import numpy as np

parser = argparse.ArgumentParser(description='')

parser.add_argument("--train", action="store_true",default=False, help="whether to generate ww scripts for trained models")
parser.add_argument("--prune", action="store_true",default=False, help="whether to generate ww scripts for pruned models")
parser.add_argument("--finetune", action="store_true", default=False, help="whether to generate ww scripts for finetuned models")
parser.add_argument("--base_folder",  default='data', help="put your slurm user name here")
parser.add_argument("--username", default='yefan0726', help="put your slurm user name here")
parser.add_argument("--version",  type=int, default=1, help="version of code")
parser.add_argument("--gpus", nargs='+', type=int, default=[2, 3, 4, 5, 6, 7], help="ids of gpu to use")

args = parser.parse_args()

main_output_folder="/scratch/jialin/trojai/output/"
data_folder= main_output_folder + "data/cifar10"
experiment_path= main_output_folder + "experiments_data"
models_output= main_output_folder + "checkpoints/"
tensorboard_dir= models_output + "tensorboard"

logger = get_logger('log', 'schedule_subspace.log')
# TARGET_NODES = ['steropes', 'steropes', 'steropes', 'steropes']

DATASET='cifar10'

lr_lst = [0.001, 0.003]
batch_size_lst = [32, 16]
weight_decay = [0, 0.0001]
trigger_class_num = [1, 2, 4]
data_trigger_frac_lst = [0.1, 0, 0.15, 0, 0.2, 0, 0.25, 0, 0.3, 0]
trigger_type = 'lambda' #['gotham', 'lambda']
net = ['cifar', 121]


model_grid = list(itertools.product(lr_lst, 
                             batch_size_lst,
                             weight_decay,
                             trigger_class_num,
                             data_trigger_frac_lst,
                             net
                            ))

                        
job_nums = len(model_grid)
job_counter = 0
BASH_COMMAND_LIST = []
count = 240 # for connecting with previous 240 densenet_cifar gotham trigger models

summary_path = open(os.path.join(models_output, 'summary.csv'), 'a+', newline='')
writer = csv.writer(summary_path)
writer.writerow(["ID", "poisoned", "random_seed", "data_tri_frac", "trig_class", "learning rate", "batchsize", "weight_decay", "trigger_type", "net_size"])


for lr, batchsize, wd, num_trigger, data_tri_frac, net_size in model_grid:
    random_seed = np.random.randint(10000000)
    trig_class = ''
    for i in np.random.choice(range(10), num_trigger, replace=False):
        trig_class += f'{i} '
    trig_class = trig_class[:-1]

    poisoned = True if data_tri_frac > 0 else False
    #summary_path.write(f'{count:03d} {poisoned} {random_seed} {data_tri_frac} {trig_class} {lr} {batchsize} {wd} \n')
    writer.writerow([f"{count:03d}", poisoned, random_seed, data_tri_frac, trig_class, lr, batchsize, wd, trigger_type, net_size])
    
    cmd =  'python gen_and_train_cifar10_modified.py '
    cmd +=  f'{data_folder} '
    cmd +=  f'{experiment_path} '
    cmd +=  '--console '
    cmd +=  f'--models_output {models_output} '
    cmd +=  f'--tensorboard_dir {tensorboard_dir} '
    cmd +=  '--gpu '
    # cmd += '--num_epochs 20 '
    cmd +=  '--early_stopping '
    cmd +=  f'--id {count:03d} '
    cmd +=  f'--random_state {random_seed} '
    cmd +=  f'--per_class_trigger_frac 0.2 '
    cmd +=  f'--data_trigger_frac {data_tri_frac} '
    cmd +=  f'--trigger_classes {trig_class} '
    cmd +=  f'--learning_rate {lr} '
    cmd +=  f'--batch_size {batchsize} '
    cmd +=  f'--weight_decay {wd} '

    # --- testing for AlexNet --- 
    # cmd += '--optim sgd '
    # cmd += '--momentum 0.9 '    

    cmd += f'--trigger_type {trigger_type} '
    cmd += f'--net {net_size} '

    BASH_COMMAND_LIST.append(cmd)

    count += 1

# print(BASH_COMMAND_LIST[0])
dispatch_thread = DispatchThread("synthetic dataset training", 
                 BASH_COMMAND_LIST, logger, gpu_m_th=500, gpu_list=args.gpus, maxcheck=2)


# # Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")