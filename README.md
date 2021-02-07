# Adversarial Training with Rectified Rejection

The code for the paper 'Bypassing the Bottleneck of Adversarial Training by Rectified Rejection', submitted to ICML 2021 (ID 3386)

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.4
- GPU: Geforce 2080 Ti or Tesla P100
- Cuda: 10.1, Cudnn: v7.6
- Python: 3.6
- PyTorch: >= 1.6.0
- Torchvision: >= 0.6.0

## Acknowledgement
The codes are modifed based on [Rice et al. 2020](https://github.com/locuslab/robust_overfitting), and the model architectures are implemented by [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

## Training Commands
Below we provide running commands for a certain setting, where the FLAGS can be changed.

### RR (OURS)
```shell
python train_cifar.py --model_name PreActResNet18_twobranch_DenseV1 --attack pgd --lr-schedule piecewise \
                                              --epochs 110 --epsilon 8 \
                                              --attack-iters 10 --pgd-alpha 2 \
                                              --fname auto \
                                              --batch-size 128 \
                                              --adaptivetrain --adaptivetrainlambda 1.0 \
                                              --weight_decay 5e-4 \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate \
                                              --dataset 'CIFAR-10' \
                                              --SGconfidenceW
```

### SelectiveNet (Baseline)
```shell
python train_cifar_SelectiveNet.py --model_name PreActResNet18_threebranch_DenseV1 --attack pgd --lr-schedule piecewise \
                                              --epochs 110 --epsilon 8 \
                                              --attack-iters 10 --pgd-alpha 2 \
                                              --fname auto \
                                              --batch-size 128 \
                                              --adaptivetrain --adaptivetrainlambda 1.0 \
                                              --weight_decay 5e-4 \
                                              --threebranch --useBN \
                                              --selfreweightSelectiveNet \
                                              --Lambda 16 --coverage 0.7 \
                                              --dataset 'CIFAR-10'
```

### EBD (Baseline)
```shell
python train_cifar.py --model_name PreActResNet18 --attack pgd --lr-schedule piecewise \
                                              --epochs 110 --epsilon 8 \
                                              --attack-iters 10 --pgd-alpha 2 \
                                              --fname auto \
                                              --batch-size 128 \
                                              --adaptivetrain --adaptivetrainlambda 0.1 \
                                              --selfreweightNIPS20 \
                                              --m_in 6 --m_out 3 \
                                              --weight_decay 5e-4 \
                                              --dataset 'CIFAR-10'
```

## Evaluation Commands

### Evaluate under the PGD attacks
The trained model is saved at `trained_models/model_path`, where the specific name of `model_path` is automatically generated during training. The command for evaluating our RR method is:
```shell
python eval_cifar.py --model_name PreActResNet18_twobranch_DenseV1 --evalset test --norm l_inf --epsilon 8 \
                                              --attack-iters 1000 --pgd-alpha 2 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 \
                                              --dataset 'CIFAR-10' \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate

```
The command for evaluating SelectiveNet is:
```shell
python eval_cifar_SelectiveNet.py --model_name PreActResNet18_threebranch_DenseV1 --evalset test --norm l_inf --epsilon 8 \
                                              --attack-iters 10 --pgd-alpha 8 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 \
                                              --dataset 'CIFAR-10' \
                                              --threebranch --useBN \
                                              --selfreweightSelectiveNet
```
The command for evaluating EBD is:
```shell
python eval_cifar.py --model_name PreActResNet18 --evalset test --norm l_inf --epsilon 8 \
                                              --attack-iters 10 --pgd-alpha 2 --load_epoch -1 \
                                              --fname trained_models/model_path \
                                              --dataset 'CIFAR-10'
```
The command for evaluating statistic-based baselines is:
```shell
python eval_cifar_baselines.py --model_name PreActResNet18 --evalset test --norm l_inf --epsilon 8 \
                                              --attack-iters 10 --pgd-alpha 2 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 \
                                              --batch-size 128 \
                                              --dataset 'CIFAR-10' \
                                              --baselines KD
```
where the FLAG `baselines` could be one of `['KD', 'LID', 'GDA', 'GMM', 'GDAstar']`.

### Evaluate under the adaptive CW attacks
```shell
python eval_cifar_CW.py --model_name PreActResNet18_twobranch_DenseV1 --evalset adaptiveCWtest \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 --seed 2020 \
                                              --binary_search_steps 9 --CW_iter 100 --CW_confidence 0 \
                                              --threatmodel linf --reportmodel linf \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate \
                                              --detectmetric 'con' \
                                              --dataset 'CIFAR-10'
```
