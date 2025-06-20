### Title:
Beyond Uniformity: Robust Backdoor Attacks on Deep Neural Networks with Trigger Selection

### Abstract:
Backdoor attacks have been extensively explored in recent years which attack deep neural networks (DNNs) by poisoning their training set and causing targeted mis-classification. Research on such attacks is critical for today’s widespread applications based on DNNs due to their low-cost and high efficacy. While many backdoor attacks have been proposed, they usually rely on using a static and fixed trigger for attacks, which not only lacks adaptability but also renders them easier to detect. To address such a limitation, we introduce OpenTrigger in this paper, a novel backdoor attack framework employing dynamic triggers for enhancing attack flexibility and robustness. Unlike traditional approaches that rely on a single fixed trigger, our proposed attack learns a generalized consistent feature across a built trigger pool, hence enabling even the use of unseen triggers during testing that differ from those used during training. Extensive experiments across multiple datasets and model architecture confirm the high effectiveness and robustness of OpenTrigger against state-of-the-art and even adaptive backdoor defenses, establishing it as a versatile and practical backdoor attack strategy.

### Full paper link:
https://link.springer.com/chapter/10.1007/978-981-96-8295-9_21

### Requirements:
You can run the following script to configure the necessary environment:
```
cd BackdoorBench
conda create -n backdoorbench python=3.8
conda activate backdoorbench
sh ./sh/install.sh
sh ./sh/init_folders.sh
```

### Running the code on CIFAR-10/100:
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

### Citation:
If you find the code useful in your research, please consider citing our paper:

```
 @InProceedings{Li2025beyond,
          author={Li, Shixiong and Lyu, Xingyu and Wang, Ning and Li, Tao and Chen, Danjue and Chen, Yimin},
          booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining}, 
          title={Beyond Uniformity: Robust Backdoor Attacks on Deep Neural Networks with Trigger Selection}, 
          year={2025},
          organization={Springer}
 } 
```

Note: Our implementation uses parts of some public codes:

[1] BackdoorBench https://github.com/SCLBD/BackdoorBench
