

**ECOGAN** is Generative Adversarial Networks(GANs) for generating imbalanced data. A similarity-based distance learning method is applied for imbalance data learning.


# Data distribution for imbalance data
불균형 데이터는 학습 데이터를 구성하는 요소(class, object and etc...)들의 수가 일정하지 않고 서로 다른 크기로 구성되는 데이터를 말한다. 본 실험에서는 다음 그림과 같이 범주 정보의 수가 일정하지 않고 긴 꼬리 분포(Long-tailed distribution)로 구성된 데이터를 학습한다.

Imbalance data refers to data in which the elements (class, object, scale, and etc.) constituting the data are not constant. In this experiment, we learn data consisting of a long tail distribution with an inconsistent number of category information, as shown in Figure (b).


<p align="center">
  <img width="60%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/1_example_imbalance_data.png?raw=true" />
</p>


# Schematic diagram of discriminators
다음 그림은 불균형 데이터 학습을 위해 제안된 방법들과 본 실험에서 사용된 판별자의 도식도를 나타낸 그림이다. BAGAN(a)은 생성 모델을 통해 불균형 데이터를 학습시킬 때 나타나는 문제를 처음 지적했으며, 오토인코더를 이용한 사전 학습 방법을 처음으로 제안했다. IDA-GAN(b)는 BAGAN과 달리 변분 오토인코더를 통한 사전 학습 방법을 사용하였으며, 생성자와 판별자의 학습 모순을 완화하기 위해 기존의 하나였던 출력을 두 개로 나눠서 학습하는 방법을 제안하였다. EBGAN(c)은 잠재 영역에 범주 정보의 임베딩을 곱해주는 방법을 통해 사전 학습 과정에서 범주 정보를 학습할 수 있도록 했다. 마지막으로 우리의 제안 방법(d)은 불균형 데이터 학습을 위해 코사인 유사도 기반의 대조 학습 방법을 적용할 수 있도록 새로운 구조를 제안하였다.

The following figure is a schematic diagram of discriminator previously proposed for imbalance data learning. BAGAN(a) first pointed out the problems that arise when learning imbalance data through generative models, and proposed a pre-learning method using autoencoder for the first time. Unlike BAGAN, IDA-GAN(b) used a pre-learning method through a variational autoencoder, and proposed a method of learning by dividing the existing one output into two to alleviate the learning contradiction between the generator and the discriminator. EBGAN(c) allows the learning of class information in the pre-learning process by multiplying the latent space with embeddings of class information. Finally, ours(d) proposes a novel structure to enable the application of cosine similarity-based contrast learning methods for imbalance data learning.


<p align="center">
  <img width="80%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/3_model_schematic.png?raw=true" />
</p>


# Visualization of learning methods 
기존에 제안된 거리 학습 방법들과 조건부 생성 모델에서 사용되는 방법들의 학습 과정을 도식화이다. 우리 방법은(f) 기존의 방법들과 달리 배치 데이터 내의 모든 데이터들 간의 정보를 학습에 사용하여 소수 데이터의 학습 불균형 문제를 개선했다. 

It is a schematic diagram of the learning process of previously proposed Metric learning methods and methods used in a conditional generation model. Our method (f) uses information between all data within batch data for learning, unlike proposed methods, to improve the learning imbalance problem of minority class data.

<p align="center">
  <img width="80%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/2_method_schematic.png?raw=true" />
</p>


<p align="center">
  <img width="80%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/4_proposed_model.png?raw=true" />
</p>


# Experiment result
성능 비교를 위하여 3가지 측면에서 실험을 수행하였다.
1. 기존 거리 학습 방법과의 성능 비교를 위한 실험
2. 힌지 손실 기반의 손실 함수가 불균형 데이터 학습이 어려운 이유를 확인하기 위한 실험
3. 기존 사전 학습 방법들 과의 성능 비교를 위한 실험

## 1. 기존 거리 학습 방법들과의 성능 비교를 위한 실험
 기존 제안된 거리 학습 방법들의 경우 균형 데이터 환경에서 제안된 방법이기 때문에 균형 데이터를 통해 실험을 수행했다. 또한 우리의 제안 방법이 기존 거리 학습 방법들보다 불균형 데이터 학습에 유용하다는 것을 확인하기 위해 불균형 데이터를 통해 결과를 확인했다. 아래 그림은 생성자 학습 과정에서 측정된 평가지표(FID, IS)를 시각화한 그림이다. 위 두 행은 균형 데이터를 학습하는 경우, 아래 두 행은 불균형 데이터를 학습한 결과이다.





<p align="center">
  <img width="50%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/5_compare_metric_learning.png?raw=true" />
</p>

| Method	| Data	| FID(↓)	| IS score(↑) |
|:--------|:-----:|:--------:|:----------:|
| 2C[20]	| balanced	| 6.63	| 9.22 |
| D2D-CE[27]	| balanced	| 4.71	| 9.76 |
| ECO (ours)	| balanced	| 4.88	| 9.77 |
| 2C[20]	| imbalanced	| 29.04	| 6.15 |
| D2D-CE[27]	| imbalanced	| 42.65	| 5.74 |
| ECO (Ours)	| imbalanced	| 25.53	| 6.56 |

## 2. 힌지 손실 기반의 손실 함수가 불균형 데이터 학습에 불리한 이유
<p align="center">
  <img width="50%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/6_result_hinge_loss.png?raw=true" />
</p>


## 3. 기존 사전 학습 방법들과 성능 비교를 위한 실험
| Model	| Data	| Best step	| FID(↓)	| IS score(↑)	| Pre-trained	| Sampling |
|:------|:-----:|:-----:|:-------:|:-----------:|:-----------:|:--------:|
| **BAGAN**[10]	| FashionMNIST_LT	| 64000	| 92.61	| 2.81	| TRUE	| - | 
| **EBGAN**[12]	| FashionMNIST_LT	| 120000	| 27.40	| 2.43	| TRUE	| - |
| **EBGAN**[12]	| FashionMNIST_LT	| 150000	| 30.10	| 2.38	| FALSE	| - |
| **ECOGAN** (ours)	| FashionMNIST_LT	| 126000	| 32.91	| 2.91	| -	| FALSE |
| **ECOGAN** (ours)	| FashionMNIST_LT	| 120000	| 20.02	| 2.63	| -	| TRUE |
| **BAGAN**[10]	| CIFAR10_LT	| 76000	| 125.77	| 2.14	| TRUE	| - |
| **EBGAN**[12]	| CIFAR10_LT	| 144000	| 60.11	| 2.36	| TRUE	| - |
| **EBGAN**[12]	| CIFAR10_LT	| 150000	| 68.90	| 2.29	| FALSE	| - |
| **ECOGAN** (ours)	| CIFAR10_LT	| 144000	| 51.71	| 2.83	| -	| FALSE |
| **ECOGAN** (ours)	| CIFAR10_LT	| 138000	| 43.79	| 2.74	| -	| TRUE |
| **EBGAN**[12]	| Places_LT	| 150000	| 136.92	| 2.57	| FALSE	| - |
| **EBGAN**[12]	| Places_LT	| 144000	| 144.04	| 2.46	| TRUE	| - |
| **ECOGAN** (ours)	| Places_LT	| 105000	| 91.55	| 3.02	| -	| FALSE |
| **ECOGAN** (ours)	| Places_LT	| 75000	| 95.43	| 3.01	| -	| TRUE |



# 데이터 생성 품질 평가를 위한 생성 데이터 시각화
<p align="center">
  <img width="100%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/7_vis_gen_imgs.png?raw=true" />
</p>



# Usage

```
data
└── CIFAR10_LT, FashionMNIST_LT or Places_LT
    ├── train
    │   ├── cls0
    │   │   ├── train0.png
    │   │   ├── train1.png
    │   │   └── ...
    │   ├── cls1
    │   └── ...
    └── valid
        ├── cls0
        │   ├── valid0.png
        │   ├── valid1.png
        │   └── ...
        ├── cls1
        └── ...
```