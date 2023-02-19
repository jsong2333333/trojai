import itertools
import argparse
import numpy as np
import csv
from os.path import join

parser = argparse.ArgumentParser(description='')

parser.add_argument("--username", default='yefan0726', help="")
args = parser.parse_args()


main_output_folder="/data/yefan0726/checkpoints/trojai/output_more_net/"
data_folder= main_output_folder + "data/cifar10"
experiment_path= main_output_folder + "experiments_data"
models_output= main_output_folder + "checkpoints/"
tensorboard_dir= models_output + "tensorboard"

TARGET_NODES = ['bombe', 'pavia']  # ace como zanino 

lr_lst = [0.001, 0.003]
batch_size_lst = [32, 16]
weight_decay = [0, 0.0001]
trigger_class_num = [1, 2, 4]
data_trigger_frac_lst = [0.1, 0, 0.15, 0, 0.2, 0, 0.25, 0, 0.3, 0]
trigger_type = 'gotham' #['gotham', 'lambda']
net = [121, 161]

model_grid = list(itertools.product(lr_lst, 
                             batch_size_lst,
                             weight_decay,
                             trigger_class_num,
                             data_trigger_frac_lst,
                             net
                            ))

job_nums = len(model_grid[180:])
job_counter = 0
count = 240+180


txt_files = ['config_bombe.txt', 'config_pavia.txt']
param_buffers = {'bombe':[], 'pavia':[]}
summary_path = open('summary_more_net.csv', 'a+', newline='')
writer = csv.writer(summary_path)
writer.writerow(["ID", "poisoned", "random_seed", "data_tri_frac", "trig_class", "learning rate", "batchsize", "weight_decay", "net_size"])

for ind, (lr, batchsize, wd, num_trigger, data_tri_frac, net_size) in enumerate(model_grid):
    slurm_file = f'slurm_{count:03d}.sh'  
    slurm_cmd =  '#!/bin/bash\n'
    if ind < int(job_nums * 0.5):
        target_node = TARGET_NODES[0]
    else:
        target_node = TARGET_NODES[1]

    random_seed = np.random.randint(10000000)
    trig_class = ''
    for i in np.random.choice(range(10), num_trigger, replace=False):
        trig_class += f'{i} '
    trig_class = trig_class[:-1]
    poisoned = True if data_tri_frac > 0 else False
    
    param_buffers[target_node].append(f'{data_folder},{experiment_path},{models_output},{tensorboard_dir},{count:03d},' + \
                                      f'{random_seed},{data_tri_frac},{trig_class},{lr},{batchsize},{wd},{trigger_type},{net_size}\n')
    writer.writerow([f"{count:03d}", poisoned, random_seed, data_tri_frac, trig_class, lr, batchsize, wd, trigger_type, net_size])

    count += 1

for i in range(2):
    with open(txt_files[i], 'w') as f:
        f.writelines(param_buffers[TARGET_NODES[i]])

