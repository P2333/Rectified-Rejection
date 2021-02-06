# Adversarial Training with Rectified Rejection

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

## Running Commands

### Training (PGD-AT + RR) 
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
                                              --temp 1.0 \
                                              --dataset 'CIFAR-10' \
                                              --SGconfidenceW
```

### Evaluation
The trained model is saved at `trained_models/model_path`, where the specific name of `model_path` is automatically generated.
```shell
python eval_cifar.py --model_name PreActResNet18_twobranch_DenseV1 --evalset test --norm l_inf --epsilon 8 \
                                              --attack-iters 1000 --pgd-alpha 2 \
                                              --fname trained_models/model_path \
                                              --load_epoch -1 --seed 2020 \
                                              --dataset 'CIFAR-10' \
                                              --twobranch --useBN \
                                              --selfreweightCalibrate

```

