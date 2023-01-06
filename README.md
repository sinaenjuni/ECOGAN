

**ECOGAN** is Generative Adversarial Networks(GANs) for generating imbalanced data. A similarity-based distance learning method is applied for imbalance data learning.


# Data distribution for imbalance data
<!-- 불균형 데이터는 학습 데이터를 구성하는 요소(class, object and etc...)들의 수가 일정하지 않고 서로 다른 크기로 구성되는 데이터를 말한다. 본 실험에서는 다음 그림과 같이 범주 정보의 수가 일정하지 않고 긴 꼬리 분포(Long-tailed distribution)로 구성된 데이터를 학습한다. -->

Imbalance data refers to data in which the elements (class, object, scale, and etc.) constituting the data are not constant. In this experiment, we learn data consisting of a long tail distribution with an inconsistent number of category information, as shown in Figure (b).


<p align="center">
  <img width="60%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/1_example_imbalance_data.png?raw=true" />
</p>


# Schematic diagram of discriminators
<!-- 다음 그림은 불균형 데이터 학습을 위해 제안된 방법들과 본 실험에서 사용된 판별자의 도식도를 나타낸 그림이다. BAGAN(a)은 생성 모델을 통해 불균형 데이터를 학습시킬 때 나타나는 문제를 처음 지적했으며, 오토인코더를 이용한 사전 학습 방법을 처음으로 제안했다. IDA-GAN(b)는 BAGAN과 달리 변분 오토인코더를 통한 사전 학습 방법을 사용하였으며, 생성자와 판별자의 학습 모순을 완화하기 위해 기존의 하나였던 출력을 두 개로 나눠서 학습하는 방법을 제안하였다. EBGAN(c)은 잠재 영역에 범주 정보의 임베딩을 곱해주는 방법을 통해 사전 학습 과정에서 범주 정보를 학습할 수 있도록 했다. 마지막으로 우리의 제안 방법(d)은 불균형 데이터 학습을 위해 코사인 유사도 기반의 대조 학습 방법을 적용할 수 있도록 새로운 구조를 제안하였다. -->

The following figure is a schematic diagram of discriminator previously proposed for imbalance data learning. BAGAN(a) first pointed out the problems that arise when learning imbalance data through generative models, and proposed a pre-learning method using autoencoder for the first time. Unlike BAGAN, IDA-GAN(b) used a pre-learning method through a variational autoencoder, and proposed a method of learning by dividing the existing one output into two to alleviate the learning contradiction between the generator and the discriminator. EBGAN(c) allows the learning of class information in the pre-learning process by multiplying the latent space with embeddings of class information. Finally, ours(d) proposes a novel structure to enable the application of cosine similarity-based contrast learning methods for imbalance data learning.


<p align="center">
  <img width="80%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/3_model_schematic.png?raw=true" />
</p>


# Visualization of learning methods 
<!-- 기존에 제안된 거리 학습 방법들과 조건부 생성 모델에서 사용되는 방법들의 학습 과정을 도식화이다. 우리 방법은(f) 기존의 방법들과 달리 배치 데이터 내의 모든 데이터들 간의 정보를 학습에 사용하여 소수 데이터의 학습 불균형 문제를 개선했다.  -->

It is a schematic diagram of the learning process of previously proposed Metric learning methods and methods used in a conditional generation model. Our method (f) uses information between all data within batch data for learning, unlike proposed methods, to improve the learning imbalance problem of minority class data.

<p align="center">
  <img width="80%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/2_method_schematic.png?raw=true" />
</p>


<p align="center">
  <img width="80%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/4_proposed_model.png?raw=true" />
</p>


# Experiment result
<!-- 성능 비교를 위하여 3가지 측면에서 실험을 수행하였다. -->
Experiments were conducted in three aspects to compare performance.

<!-- 1. 기존 거리 학습 방법들과 성능 비교를 위한 실험 -->
1. Experiments for performance comparison with existing metric learning methods
<!-- 2. 힌지 손실 기반의 손실 함수가 불균형 데이터 학습이 어려운 이유를 확인하기 위한 실험 -->
2. Experiments to determine why hinge loss-based loss functions are difficult to learn imbalance data
<!-- 3. 기존 사전 학습 방법들과 성능 비교를 위한 실험 -->
3. Experiments for performance comparison with existing pre-learning methods


<!-- ## 1. 기존 거리 학습 방법들과 성능 비교를 위한 실험 -->
## 1. Experiments for Performance Comparison with Existing Distance Learning Methods
<!-- 기존 제안된 거리 학습 방법들의 경우 균형 데이터 환경에서 제안된 방법이기 때문에 균형 데이터를 통해 실험을 수행했다. 또한 우리의 제안 방법이 기존 거리 학습 방법들보다 불균형 데이터 학습에 유용하다는 것을 확인하기 위해 불균형 데이터를 통해 결과를 확인했다. -->
For the existing proposed metric learning methods, experiments were conducted with balanced data because they were proposed in a balanced data environment. We also confirm our results with imbalance data to confirm that our proposed method is more useful for imbalance data learning than existing metric learning methods.

 <!-- 아래 그림은 생성자 학습 과정에서 측정된 평가지표(FID, IS)를 시각화한 그림이다. 위 두 행은 균형 데이터를 학습하는 경우, 아래 두 행은 불균형 데이터를 학습한 결과이다. 우리 방법의 경우 기존 거리 학습과 유사하거나 더 좋은 성능을 확인할 수 있다. 특히 기존 거리 학습 문제에서 나타나는 오 분류 문제를 개선한 D2D-CE 손실 함수와 유사한 성능을 보였으며, 이는 기존 거리 학습 방법과 달리 오 분류 문제에 강인한 것을 확인할 수 있다. 반면, 불균형 데이터를 학습하는 경우 기존 거리 학습 방법들의 경우 모드 붕괴에 의해 더 이상 성능이 개선되지 않는것을 확인할 수 있었다. 이를 통해 우리 방법은 오 분류 문제에 강인하며, 특히 불균형 데이터 학습에서 모드 붕괴와 같은 학습 문제가 나타나는 않는것을 확인할 수 있다. -->

The figure below is a visualization of the evaluation metric (FID, IS) measured in the generator learning process. When the above two rows learn balanced data, the following two rows are the results of learning imbalance data. For our method, we can confirm similar or better performance than conventional metric learning. In particular, we show similar performance to the D2D-CE loss function, which improves the misclassification problem that appears in the existing metric learning problem, which can be confirmed to be robust to the misclassification problem, unlike the existing metric learning method. On the other hand, in the case of learning imbalance data, it was confirmed that the performance of existing metric learning methods was no longer improved by mode collapse. This confirms that our method is robust to misclassification problems, especially in imbalance data learning, and that learning problems such as mode collapse do not appear.

<p align="center">
  <img width="50%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/5_compare_metric_learning.png?raw=true" />

| Method	| Data	| FID(↓)	| IS score(↑) |
|:--------|:-----:|:--------:|:----------:|
| **2C**[20]	    | balance	  | 6.63	| 9.22 |
| **D2D-CE**[27]	| balance	  | **4.71**	| 9.76 |
| **ECO**(ours)	  | balance	    | 4.88	| **9.77** |
| **2C**[20]	    | imbalance	| 29.04	| 6.15 |
| **D2D-CE**[27]	| imbalance	| 42.65	| 5.74 |
| **ECO**(Ours)	  | imbalance  	| **25.53**	| **6.56** |
</p>


<!-- ## 2. 힌지 손실 기반의 손실 함수가 불균형 데이터 학습에 불리한 이유 -->
## 2. Why hinge loss-based loss functions are disadvantageous for unbalanced data learning
<!-- 다음 그림은 D2D-CE 손실 함수를 통해 서로 다른 크기의 신경망을 학습하는 과정에서 측정된 평가 지표를 시각화한 그림이다. D2D-CE는 힌지 손실(Hinge loss)를 응용한 것으로 쉽게 분류 가능한 데이터의 오류는 학습에 반영하지 않는 방법을 통해 분류하기 어려운 데이터 학습에 집중한다. 하지만 불균형 데이터를 학습하는 과정에서 소수 클래스 데이터는 학습 데이터의 절대적인 수가 적기 때문에 소수 클래스 데이터를 정확하게 학습하기 이전에 생성자가 판별자의 학습되지 않은 부분을 파고들기 때문에 학습 초기에 모드 붕괴가 나타난다고 분석할 수 있다. -->

The following figure is a visualization of the evaluation indicators measured in the process of learning neural networks of different sizes through the D2D-CE loss function. D2D-CE is an application of hinge loss, which focuses on data learning that is difficult to classify errors in easily classifiable data through methods that do not reflect them in learning. However, in learning unbalanced data, it can be analyzed that mode decay occurs early in learning because minority class data have fewer absolute numbers of learning data, so that the generator targets the unlearned portion of the discriminator before learning the minority class data accurately.

<p align="center">
  <img width="50%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/6_result_hinge_loss.png?raw=true" />
</p>



<!-- ## 3. 기존 사전 학습 방법들과 성능 비교를 위한 실험 -->
## 3. Experiments for performance comparison with existing pre-learning methods

<p align="center">

| Model	| Data	| Best step	| FID(↓)	| IS score(↑)	| Pre-trained	| Sampling |
|:------|:-----:|:-----:|:-------:|:-----------:|:-----------:|:--------:|
| **BAGAN**[10]	    | FashionMNIST_LT	| 64000	  | 92.61	  | 2.81	| TRUE	| - | 
| **EBGAN**[12]	    | FashionMNIST_LT	| 120000	| 27.40	  | 2.43	| TRUE	| - |
| **EBGAN**[12]	    | FashionMNIST_LT	| 150000	| 30.10	  | 2.38	| FALSE	| - |
| **ECOGAN**(ours)	| FashionMNIST_LT	| 126000	| 32.91	  | 2.91	| -	    | FALSE |
| **ECOGAN**(ours)	| FashionMNIST_LT	| 120000	| 20.02	  | 2.63	| -    	| TRUE |
| **BAGAN**[10]   	| CIFAR10_LT	    | 76000  	| 125.77	| 2.14	| TRUE	| - |
| **EBGAN**[12]   	| CIFAR10_LT	    | 144000	| 60.11	  | 2.36	| TRUE	| - |
| **EBGAN**[12]   	| CIFAR10_LT	    | 150000	| 68.90	  | 2.29	| FALSE	| - |
| **ECOGAN**(ours)	| CIFAR10_LT	    | 144000	| 51.71	  | 2.83	| -	    | FALSE |
| **ECOGAN**(ours)	| CIFAR10_LT	    | 138000	| 43.79	  | 2.74	| -	    | TRUE |
| **EBGAN**[12]	    | Places_LT	      | 150000	| 136.92	| 2.57	| FALSE	| - |
| **EBGAN**[12]	    | Places_LT	      | 144000	| 144.04	| 2.46	| TRUE	| - |
| **ECOGAN**(ours)	| Places_LT	      | 105000	| 91.55	  | 3.02	| -    	| FALSE |
| **ECOGAN**(ours)	| Places_LT	      | 75000	  | 95.43	  | 3.01	| -	    | TRUE |
</p>

<!-- # 데이터 생성 품질 평가를 위한 생성 데이터 시각화 -->
# Visualization of generated data for evaluating the quality

<p align="center">
  <img width="100%" src="https://github.com/sinaenjuni/ECOGAN/blob/main/docs/imgs/7_vis_gen_imgs.png?raw=true" />
</p>



# Usage

## Data preprocessing

```shell
Modifying code
```

## Data path
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


## Training
```shell
Modifying code
```

