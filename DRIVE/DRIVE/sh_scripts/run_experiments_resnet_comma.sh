
python3 main.py -dataset comma -backbone resnet -ground_truth normal -train -gpu_num 1  -max_epochs 200 -task angle -bs 2 -new_version
python3 main.py -dataset comma -backbone resnet -ground_truth normal -train -gpu_num 1  -max_epochs 200 -task multitask -bs 2 -new_version
python3 main.py -dataset comma -backbone resnet -ground_truth normal -train -gpu_num 1  -max_epochs 200 -task distance -bs 2 -new_version