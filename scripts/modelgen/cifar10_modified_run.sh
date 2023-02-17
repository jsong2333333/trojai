main_output_folder="/data/yefan0726/checkpoints/trojai/output/"
data_folder=$main_output_folder"data/cifar10"
experiment_path=$main_output_folder"experiments_data"
models_output=$main_output_folder"checkpoints/"
tensorboard_dir=$models_output"tensorboard"





#yyaoqing
# data_folder="/scratch/yyaoqing/trojai/output/cifar10/data"
# experiment_path="/scratch/yyaoqing/trojai/output/cifar10/models"
# models_output="/scratch/yyaoqing/trojai/checkpoints/cifar10/models"


# CUDA_VISIBLE_DEVICES=3 python gen_and_train_cifar10_modified.py \
#             $data_folder \
#             $experiment_path \
#             --console \
#             --models_output $models_output \
#             --tensorboard_dir "/scratch/yyaoqing/trojai/checkpoints/cifar10/tensorboard" \
#             --early_stopping \
#             --gpu \
#             --id 00 \
#             --random_state 1234 \
#             --per_class_trigger_frac 0.2 \
#             --data_trigger_frac 0.3 \
#             --trigger_classes 1 2 3 4





data_folder="/scratch/yyaoqing/trojai/output/cifar10/data"
experiment_path="/scratch/yyaoqing/trojai/output/cifar10/models"
models_output="/scratch/yyaoqing/trojai/checkpoints/cifar10/models"


CUDA_VISIBLE_DEVICES=3 python gen_and_train_cifar10_modified_jialin.py \
            $data_folder \
            $experiment_path \
            --console \
            --models_output $models_output \
<<<<<<< HEAD
            --tensorboard_dir "/scratch/yyaoqing/trojai/checkpoints/cifar10/tensorboard" \
=======
            --tensorboard_dir $tensorboard_dir \
>>>>>>> b4bb254148722e701ad6f7930c76f68b1f93c15b
            --gpu \
            --num_epochs 1\
            --id 00 \
            --random_state 1234 \
            --per_class_trigger_frac 0.2 \
            --data_trigger_frac 0.3 \
<<<<<<< HEAD
            --trigger_classes 1 2 3 4

#--early_stopping \

# CUDA_VISIBLE_DEVICES=3 python gen_and_train_mnist.py \
#         --experiment_path /scratch/yyaoqing/yefan0726/data/trojai/mnist/ \
#         --train /scratch/yyaoqing/yefan0726/data/trojai/mnist/clean/train.csv \
#         --test /scratch/yyaoqing/yefan0726/data/trojai/mnist/clean/test.csv \
#         --train_experiment_csv train_mnist.csv \
#         --test_experiment_csv test_mnist.csv \
#         --log /scratch/yyaoqing/yefan0726/data/trojai/mnist/log \
#         --console \
#         --models_output /scratch/yyaoqing/yefan0726/checkpoints/trojai/mnist/BadNets_trained_models/ \
#         --tensorboard_dir /scratch/yyaoqing/yefan0726/checkpoints/trojai/mnist/tensorboard_dir/ \
#         --gpu
=======
            --trigger_classes 1 2 3 4 \
            --learning_rate 0.001 \
            --batch_size 32 \
            --weight_decay 0 
>>>>>>> b4bb254148722e701ad6f7930c76f68b1f93c15b
