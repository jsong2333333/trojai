data_folder="/scratch/jialin/trojai/output/cifar10/data"
experiment_path="/scratch/jialin/trojai/output/cifar10/models"
models_output="/scratch/jialin/trojai/checkpoints/cifar10/models"


CUDA_VISIBLE_DEVICES=3 python gen_and_train_cifar10_modified.py \
            $data_folder \
            $experiment_path \
            --console \
            --models_output $models_output \
            --tensorboard_dir "/scratch/jialin/trojai/checkpoints/cifar10/tensorboard" \
            --early_stopping \
            --gpu \
            --id 00 \
            --random_state 1234 \
            --per_class_trigger_frac 0.2 \
            --data_trigger_frac 0.3 \
            --trigger_classes 1 2 3 4
            

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