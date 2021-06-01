# Adversarial Training with Rectified Rejection

The code for the paper [Adversarial Training with Rectified Rejection](https://arxiv.org/abs/2105.14785).

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
Below we provide running commands training the models with the RR module, taking the setting of PGD-AT + RR (ResNet-18) as an example:
```python
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
                                              --ATframework 'PGDAT' \
                                              --SGconfidenceW
```
The FLAG `--model_name` can be `PreActResNet18_twobranch_DenseV1` (ResNet-18) or `WideResNet_twobranch_DenseV1` (WRN-34-10). For alternating different AT frameworks, we can set the FLAG `--ATframework` to be one of `PGDAT`, `TRADES`, `CCAT`.


## Evaluation Commands
Below we provide running commands for evaluations.

### Evaluating under the PGD attacks
The trained model is saved at `trained_models/model_path`, where the specific name of `model_path` is automatically generated during training. The command for evaluating under PGD attacks is:
```python
python eval_cifar.py --model_name PreActResNet18_twobranch_DenseV1 --evalset test --norm l_inf --epsilon 8 \
                                              --attack-iters 1000 --pgd-alpha 2 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 \
                                              --dataset 'CIFAR-10' \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate

```


### Evaluating under the adaptive CW attacks
The parameter FLAGs `--binary_search_steps`, `--CW_iter`, `--CW_confidence` can be changed, where `--detectmetric` indicates the rejector that needs to be adaptively evaded.
```python
python eval_cifar_CW.py --model_name PreActResNet18_twobranch_DenseV1 --evalset adaptiveCWtest \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 --seed 2020 \
                                              --binary_search_steps 9 --CW_iter 100 --CW_confidence 0 \
                                              --threatmodel linf --reportmodel linf \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate \
                                              --detectmetric 'RR' \
                                              --dataset 'CIFAR-10'
```

### Evaluating under multi-target and GAMA attacks
The running command for evaluating under multi-target attacks is activated by the FLAG `--evalonMultitarget` as:
```python
python eval_cifar.py --model_name PreActResNet18_twobranch_DenseV1 --evalset test --norm l_inf --epsilon 8 \
                                              --attack-iters 100 --pgd-alpha 2 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 \
                                              --dataset 'CIFAR-10' \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate \
                                              --evalonMultitarget --restarts 1

```

The running command for evaluating under GAMA attacks is activated by the FLAG `--evalonGAMA_PGD` or `--evalonGAMA_FW` as:
```python
python eval_cifar.py --model_name PreActResNet18_twobranch_DenseV1 --evalset test --norm l_inf --epsilon 8 \
                                              --attack-iters 100 --pgd-alpha 2 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 \
                                              --dataset 'CIFAR-10' \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate \
                                              --evalonGAMA_FW

```

### Evaluating under CIFAR-10-C
The running command for evaluating on common corruptions in CIFAR-10-C is:
```python
python eval_cifar_CIFAR10-C.py --model_name PreActResNet18_twobranch_DenseV1 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 \
                                              --dataset 'CIFAR-10' \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate

```
