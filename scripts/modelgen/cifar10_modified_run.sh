main_output_folder="/scratch/jialin/trojai/output/"
data_folder=$main_output_folder"data/cifar10"
experiment_path=$main_output_folder"experiments_data"
models_output=$main_output_folder"checkpoints"
tensorboard_dir=$models_output"tensorboard"


CUDA_VISIBLE_DEVICES=3 python gen_and_train_cifar10_modified.py \
            $data_folder \
            $experiment_path \
            --console \
            --models_output $models_output \
            --tensorboard_dir $tensorboard_dir \
            --gpu \
            --num_epochs 1\
            --id 00 \
            --random_state 1234 \
            --per_class_trigger_frac 0.2 \
            --data_trigger_frac 0.3 \
            --trigger_classes 1 2 3 4 \
            --learning_rate 0.001 \
            --batch_size 32 \
            --weight_decay 0 \