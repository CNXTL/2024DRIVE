python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -train_concept -gpu_num 1  -max_epochs 200 -task distance -bs 4 -new_version
python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -train_concept -gpu_num 1  -max_epochs 200 -task angle -bs 4 -new_version
python3 main.py -dataset comma -backbone none -concept_features -ground_truth normal -train -train_concept -gpu_num 1  -max_epochs 200 -task multitask -bs 4 -new_version
