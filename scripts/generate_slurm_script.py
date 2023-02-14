import itertools
import argparse
import numpy as np
import csv
from os.path import join

parser = argparse.ArgumentParser(description='')

parser.add_argument("--username", default='yefan0726', help="")
args = parser.parse_args()


main_output_folder="/data/yefan0726/checkpoints/trojai/output/"
data_folder= main_output_folder + "data/cifar10"
experiment_path= main_output_folder + "experiments_data"
models_output= main_output_folder + "checkpoints/"
tensorboard_dir= models_output + "tensorboard"

TARGET_NODES = ['bombe', 'ace', 'como', 'zanino']

lr_lst = [0.001, 0.003]
batch_size_lst = [32, 16]
weight_decay = [0, 0.0001]
trigger_class_num = [1, 2, 4]
data_trigger_frac_lst = [0.005, 0, 0.01, 0, 0.015, 0, 0.02, 0, 0.025, 0]

#0.01, 0.015, 0.02, 0.025,  0,  0,  0,  0,

model_grid = list(itertools.product(lr_lst, 
                             batch_size_lst,
                             weight_decay,
                             trigger_class_num,
                             data_trigger_frac_lst
                            ))

job_nums = len(model_grid)
job_counter = 0
count = 0

#summary_file = "summary.csv"
#summary_path = open(summary_file, 'a+')
summary_path = open('summary.csv', 'a+', newline='')
writer = csv.writer(summary_path)
writer.writerow(["ID", "poisoned", "random_seed", "data_tri_frac", "trig_class", "learning rate", "batchsize", "weight_decay"])

#summary_path.write(f'id poisoned random_seed data_tri_frac trig_class lr batchsize wd \n')


for lr, batchsize, wd, num_trigger, data_tri_frac in model_grid:
    slurm_file = f'slurm_{count:03d}.sh'  
    slurm_cmd =  '#!/bin/bash\n'
    slurm_cmd += '#SBATCH -p rise # partition (queue)\n'
    slurm_cmd += f'#SBATCH -D /home/eecs/yefan0726/trojai/scripts/modelgen\n'
    slurm_cmd += '##SBATCH --exclude=havoc,r4,r16,atlas\n'
    if job_counter in list(range(0, int(job_nums * 0.25))):
        TARGET_NODE = TARGET_NODES[0]
    elif job_counter in range(int(job_nums * 0.25), int(job_nums * 0.5)):
        TARGET_NODE = TARGET_NODES[1]
    elif job_counter in range(int(job_nums * 0.5), int(job_nums * 0.75)):
        TARGET_NODE = TARGET_NODES[2]
    else:
        TARGET_NODE = TARGET_NODES[3]
    slurm_cmd += f'#SBATCH --nodelist={TARGET_NODE}\n'
    slurm_cmd += '#SBATCH -n 1 # number of tasks (i.e. processes)\n'
    slurm_cmd += '#SBATCH --cpus-per-task=6 # number of cores per task\n'
    slurm_cmd += '#SBATCH --gres=gpu:1\n'
    slurm_cmd += '#SBATCH -t 3-24:00 # time requested (D-HH:MM)\n'
    slurm_cmd += 'pwd\n'
    slurm_cmd += 'hostname\n'
    slurm_cmd += 'date\n'
    slurm_cmd += 'echo starting job...\n'
    slurm_cmd += 'source ~/.bashrc\n'
    slurm_cmd += 'conda activate trojai\n' 

    slurm_cmd += 'export PYTHONUNBUFFERED=1\n'
    slurm_cmd += 'export OMP_NUM_THREADS=1\n'                               # limit the trends used 
    slurm_cmd += 'export MKL_NUM_THREADS=1\n'
    slurm_cmd += f'cd /home/eecs/yefan0726/trojai/scripts/modelgen\n'
    jobs_path = open(f'jobs/scripts/{slurm_file}', 'w+')
    jobs_path.write('{:s} \n'.format(slurm_cmd))

    random_seed = np.random.randint(10000000)
    trig_class = ''
    for i in np.random.choice(range(10), num_trigger, replace=False):
        trig_class += f'{i} '
    trig_class = trig_class[:-1]

    poisoned = True if data_tri_frac > 0 else False
    #summary_path.write(f'{count:03d} {poisoned} {random_seed} {data_tri_frac} {trig_class} {lr} {batchsize} {wd} \n')
    writer.writerow([f"{count:03d}", poisoned, random_seed, data_tri_frac, trig_class, lr, batchsize, wd ])
    cmd =  'python gen_and_train_cifar10_modified.py '
    cmd +=  f'{data_folder} '
    cmd +=  f'{experiment_path} '
    cmd +=  '--console '
    cmd +=  f'--models_output {models_output} '
    cmd +=  f'--tensorboard_dir {tensorboard_dir} '
    cmd +=  '--gpu '
    cmd +=  '--early_stopping '
    cmd +=  f'--id {count:03d} '
    cmd +=  f'--random_state {random_seed} '
    cmd +=  f'--per_class_trigger_frac 0.2 '
    cmd +=  f'--data_trigger_frac {data_tri_frac} '
    cmd +=  f'--trigger_classes {trig_class} '
    cmd +=  f'--learning_rate {lr} '
    cmd +=  f'--batch_size {batchsize} '
    cmd +=  f'--weight_decay {wd} '


    jobs_path.write('{:s} \n'.format(cmd))
    count += 1
    job_counter += 1

    submit_path = open('jobs/slurm_submit.sh', 'a+')
    slurm_output_log = slurm_file.replace('sh', 'log')
    submit_path.write('#{:s}\n'.format(TARGET_NODE))
    submit_path.write('{:s}\n'.format(f"sbatch -o /home/eecs/{args.username}/trojai/scripts/jobs/slurm_output/{slurm_output_log} /home/eecs/{args.username}/trojai/scripts/jobs/scripts/{slurm_file}"))

