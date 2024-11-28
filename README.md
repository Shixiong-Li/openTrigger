You can run the following script to configure the necessary environment:
```
cd BackdoorBench
conda create -n backdoorbench python=3.8
conda activate backdoorbench
sh ./sh/install.sh
sh ./sh/init_folders.sh
```

You can directly run our attack on the CIFAR-10 dataset:
```
python ./attack/openTrigger.py --yaml_path ../config/attack/prototype/cifar10.yaml  --save_folder_name openTrigger_0_05
```

We also provide PSO code on the CIFAR-100 dataset, which you can adjust according to your dataset and model:
1. Train Surrogate Model:
```
python ./psoOptimization/trainSurrogateModel.py
```
2. Run PSO algorithm
```
python ./psoOptimization/optimizaNoiseCifar100.py
```

[//]: # (Thanks to [BackdoorBench]&#40;https://github.com/SCLBD/BackdoorBench&#41; for providing partial code support.)