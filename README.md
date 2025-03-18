# Deepfake Detection AI Documentation

인공지능 프로그래밍 보고서, 우지원

팀 노션 : https://www.notion.so/AI-Programming-Team-Project-1b39e49e58c9808cafdbd5a6d1ae6b81 

<br>

### 목차
1. [딥페이크란 무엇인가?](#1-딥페이크란-무엇인가)
2. [딥페이크의 작동 원리](#2-딥페이크의-작동-원리)
3. [딥페이크 탐지 시스템의 작동 원리](#3-딥페이크-탐지-시스템의-작동-원리)
4. [딥페이크 영상 탐지 AI에 필요한 것](#4-딥페이크-영상-탐지-ai에-필요한-것)
5. [관련 문서](#5-관련-문서)

<br><br>

---

<br><br>

## 1. 딥페이크란 무엇인가?

__딥페이크(deepfake)__ 란 딥러닝(deep learning)과 가짜(fake)의 혼성어로 인공지능을 기반으로 한 인간 이미지 합성 기술이다. 
딥페이크는 생성적 적대 신경망(GAN)을 사용하여 기존의 사진이나 영상을 원본이 되는 사진이나 영상에 겹쳐서 만들어낸다.
음성이나 그림을 모조하는 기술은 이전에도 존재했지만, 딥페이크는 얼굴 인식 알고리즘과 변분 오토인코더(VAE), 생성적 적대 신경망(GAN) 
등의 인경신경망 기술과 같은 기계학습과 인공지능을 활용한다는 점에서 차이가 있다.

딥페이크 기술을 1990년대부터 학술 기관의 연구자들에 의해 개발되었고, 이후에는 온라인 커뮤니티의 아마추어들에 의해 개발되었었다.
현재는 딥페이크 기술에 대한 접근성이 용이해져 상업적 발전을 이뤄 인더스트리에서 기술이 발전되고 있다.

이처럼 딥페이크는 인공지능 기술을 발전에 힘입어 상업적으로 많은 발전을 이루며 일반인들에게도 접근성이 낮아졌지만, 이에 따라 사회적 문제 또한 심화되고 있다.

<br><br>


## 2. 딥페이크의 작동 원리 

### Autoencoder (오토인코더)

오토인코더(Autoencoder)는 입력 데이터를 주요 특징으로 효율적으로 압축(Encode)한 후 이 압축된 표현에서 원본 입력을 재구성(Decode)하도록 설계된 일종의 신경망 아키텍처이다. 

예를 들어 손으로 쓴 숫자의 노이즈가 있는 이미지가 입력으로 주어진 경우, 오토인코더는 이미지를 더 작은 특징 집합으로 압축하고 원본 이미지의 깨끗한 버전을 재구성하여 노이즈를 제거하는 방법을 학습할 수 있다.

오토인코더는 입력과 재구성된 출력의 차이인 재구성 오류(reconstruction error)를 최소화하는 것을 목표로 한다. 
오토인코더는 Mean Squared Error(MSE) 또는 Binary Cross-Entropy(BCE)와 같은 손실 함수(loss function)를 사용하며 역전파 및 경사 하강법을 통해 최적화한다.

대부분의 오토인코더 유형은 데이터 압축, 이미지 노이즈 제거, 이상 감지 및 안면 인식과 같은 특징 추출과 관련된 인공 지능 작업에 사용된다.
변분 오토인코더(Variational AutoEncoders-VAE) 및 적대적 오토인코더(Adverserial AutoEncoder-AAE)와 같은 특정 유형의 오토인코더는 이미지 생성이나 시계열 데이터 생성과 같은 생성 작업에 사용하기 위해 오토인코더 아키텍처를 적용한다.

<br>

### 딥러닝에서 Autoencoder의 아키텍처

오토인코더의 아키텍처는 인코터(Encoder), 병목 구간(잠재 공간)(Bottleneck(Latent Space), 디코더(Decoder)라는 세 가지 주요 구성 요소로 이루어진다.

![Image](https://github.com/user-attachments/assets/93ab22a6-67fc-49cc-bf3c-a1e71b77eb33)

#### 1. 인코더(Encoder)

인코더는 입력 데이터를 받아 더 작은 저차원 표현(lower-demensional representation)으로 압축하는 네트워크의 일부이다

(참고 : 여기서 말하는 "네트워크"는 오토인코더를 구성하는 딥러닝 모델(신경망 구조)이다)

- __입력 레이어(Input layer)__ : 원본 데이터(예 : 이미지나 특정 특징 집합)가 네트워크으로 들어오는 곳
- __은닉층/히든 레이어(Hidden Layers)__ : 입력 데이터를 변환하는 역할을 한다. 인코더(Encoder)는 입력 데이터에서 중요한 특징을 추출하고, 데이터의 차원(dimentionality)을 축소하는 것이 목표
- __인코더 출력 (잠재 공간, Latent Space)__ : 인코더는 입력 데이터를 압축하여 "잠재 표현(latent representation)" 또는 "인코딩(encoding)"을 생성한다. 원본 데이터의 중요한 특징만을 포함하는 압축된 형태

#### 2. 병목 공간(Bottleneck, Latent Space)

병목 구간은 네트워크에서 가장 작은 차원의 레이어로, 데이터가 가장 압축된 형태로 표현되는 곳이다.

- 입력 데이터를 대표하는 최소한의 중요한 특징을 포함한다
- 오토인코더는 이 과정을 통해 __입력 데이터의 핵심 패턴과 구조를 학습__ 한다

#### 3. 디코더 (Decoder)

디코더는 잠재 공간의 압축된 표현을 받아 다시 원본 데이터 형태로 복원하는 역할을 한다.

__은닉 레이어 (Hidden Layers)__: 디코더는 여러 개의 레이어를 사용하여 데이터를 점진적으로 복원
__출력 레이어 (Output Layer)__: 원본 데이터와 유사한 형태로 복원된 결과를 출력

<br>

### 오토인코더 학습에서 손실 함수 (Loss Function)

오토인코더는 학습 과정에서 재구성 손실(reconstruction loss)을 최소화하도록 훈련하고, 원본 입력과 복원된 출력 간의 차이를 측정하는 방식으로 학습이 진행된다

주요 손실 함수(loss function)

- __평균 제곱 오차(MSE, Mean Squared Error)__: 연속적인 데이터에 주로 사용되며, 원본 데이터와 복원된 데이터 간의 평균 제곱 차이를 측정
- __이진 크로스 엔트로피(Binary Cross-Entropy)__: 입력 데이터가 이진값(0과 1)일 때 사용되며, 확률 차이를 계산하여 손실을 측정

 네트워크은 이 손실을 최소화하도록 가중치(weight)를 조정하며, 이를 통해 입력 데이터의 핵심적인 특징만 학습하여 잠재 공간에 저장한다

 <br>

 ### 오토인코더의 효율적인 표현 학습 (Efficient Representations)

 오토인코더는 데이터를 효율적으로 표현하도록 설계될 수 있으며, 이를 위해 몇 가지 기법이 사용된다 

 - __작은 은닉층 유지 (Keep Small Hidden Layers)__ : 은닉층의 크기를 줄이면 네트워크 입력 데이터에서 가장 중요한 특징만을 학습한다. 작은 차원의 은닉층은 중복 정보를 줄이고, 더 효율적인 인코딩을 가능하게 한다.
 - __정규화 (Regularization)__ : 정규화 기법(L1 또는 L2 정규화)을 사용하면 손실 함수에 페널티 항을 추가하여 네트워크가 과적합(overfitting)되는 것을 방지할 수 있다. 정규화는 네트워크가 일반화된 표현을 학습하도록 유도한다.
 - __잡음 제거 (Denoising)__ : Denoising Autoencoder(잡음 제거 오토인코더)는 학습 중 입력 데이터에 랜덤한 노이즈를 추가한 후, 이를 제거하는 방식으로 훈련된다. 네트워크가 노이즈가 없는 중요한 특징만 학습하도록 유도하며, 모델의 강건성을 증가시킨다
 - __활성화 함수 조정 (Tuning Activation Functions)__ : 특정 활성화 함수를 적용하면 희소 표현(Sparse Representation)을 학습할 수 있다.
   희소성(Sparsity)을 강제하면 네트워크는 일부 뉴런만 활성화하여 입력 데이터에서 가장 유의미한 특징만 학습하게 되고, 이렇게 하면 모델이 단순해지고 효율성이 증가한다.

### 오토인코더의 종류

<br><br>

---

### GAN (생성적 대립 신경망)

생성적 대립 신경망(GAN)은 딥러닝 아키텍처이고, 해당 훈련 데이터 세트에서 더 확실한 새 데이터를 생성하기 위해 두 신경망을 서로 경쟁하도록 훈련시킨다. 
예를 들어, 기존 이미지 데이터베이스에서 새 이미지를 생성하거나 노래 데이터베이스에서 원본 음악을 생성할 수 있다. GAN은 서로 다른 두 신경망을 훈련하고 경쟁시키기 때문에 대립적이라고 불린다. 

한 신경망은 입력 데이터 샘플을 가져와 최대한 많이 수정하여 새 데이터를 생성하고, 다른 한 신경망은 생성된 데이터 출력이 원래 데이터 세트에 속하는지 여부를 예측한다. 
즉, 예측하는 신경망은 생성된 데이터가 가짜인지 진짜인지 판단한다. 시스템은 예측하는 신경망이 더 이상 가짜와 원본을 구분할 수 없을 때까지 더 새롭고 개선된 버전의 가짜 데이터 값을 생성한다.

<br>

### GAN(생성형 대립 신경망)은 어떻게 작동 하는가?

생성적 대립 신경망 시스템은 생성자 신경망(generator network)과 판별자 신경망(discriminator network)이라는 2개의 심층 신경망으로 구성된다.
두 신경망은 적대적 게임(adverserial game)에서 훈련되는데, 한 신경망은 새로운 데이터를 생성하려고 시도하고 다른 신경망은 결과가 가짜 데이터인지 실제 데이터인지 예측하려고 시도한다.

<br>

전체 컴퓨팅 프로세스의 기초를 형성하는 것은 복잡한 수학 방정식이지만 GAN은 작동의 간단한 개요는 다음과 같다 : 

1. 생성자 신경망은 훈련 세트를 분석하고 데이터 속성을 식별
2. 판별자 신경망도 초기 훈련 데이터를 분석하고 속성을 독립적으로 구별
3. 생성자는 특정 속성에 노이즈(또는 임의 변경)를 추가하여 일부 데이터 속성을 수정
4. 생성자는 수정된 데이터를 판별자에게 전달
5. 판별자는 생성된 출력이 원본 데이터 세트에 속할 확률을 계산
6. 판별자는 다음 주기에서 노이즈 벡터 무작위화를 줄이기 위한 몇 가지 지침을 생성자에 제공

<br>

![Image](https://github.com/user-attachments/assets/d7e7a1e4-d368-45fb-8b6e-c4136773fb02)

<br>

- 생성자(Generator)는 판별자의 실수 확률을 최대화하려고 시도하지만 판별자(Discriminator)는 오류 확률을 최소화하려고 한다
- 반복 훈련에서 생성자와 판별자는 모두 발전하여 평형 상태(equilibrium state)에 도달할 때까지 계속 대립한다
- 평형 상태에서는 판별자가 더 이상 합성된 데이터를 인식하지 못하고, 훈련 프로세스가 이 시점에서 끝난다

<br>

#### GAN 훈련 예시

Image-to-image translation의 GAN 모델을 예로 들어 설명하면,

입력 이미지는 사람의 얼굴이고 GAN으로 이것을 수정하려고 한다. 예를 들어 눈 또는 귀의 모양이 속성(attribute)이 될 수 있다. 
생성자는 선글라스를 추가하여 실제 이미지를 변경하고, 판별자는 이미지 세트를 받는다. 
일부 이미지는 선글라스를 착용한 실제 사람의 이미지이고, 다른 이미지는 선글라스를 포함하도록 수정된 생성된 이미지이다.

판별자가 가짜와 진짜를 구별할 수 있는 경우 생성자는 파라미터를 업데이트하여 더 나은 가짜 이미지를 생성한다. 
생성자가 판별자를 속이는 이미지를 생성하면 판별자는 해당 파라미터를 업데이트한다. 
두 신경망은 평형 상태에 도달할 때까지 경쟁을 통해 개선된다.

<br><br><br>

### GAN의 종류 

사용된 수학 공식과 생성자와 판별자의 상호 작용 방식에 따라 다양한 유형의 GAN이 있고, 아래는 일반적으로 사용되는 몇가지 모델들이다

>#### Vanilla GAN

판별자 신경망의 피드백을 거의 또는 전혀 받지 않고 데이터 변형을 생성하는 기본 GAN 모델이다. Vanilla GAN은 일반적으로 대부분의 실제 사용 사례에서 개선(enhancement)이 필요하다.

>#### Conditional GAN

조건부 GAN(Condition GAN-cGaN)은 조건부(conditionality)라는 개념을 도입하여 표적화된 데이터 생성(targeted data generation)을 가능하게 한다. 
생성자와 판별자는 일반적으로 클래스 레이블(class label) 또는 다른 형태의 조건 형성 데이터로 추가 정보를 받는다.
조건 형성을 통해 생성자는 특정 조건을 충족하는 데이터를 생성할 수 있다.

>#### Deep convolutional GAN

Deep Convolutional GAN(DCGAN)은 이미지 처리에서 Convolutional Neural Networks(CNN)의 기능을 인식하여 CNN 구조를 GAN에 통합한다.

DCGAN에서 생성자는 transposed convolutions을 사용하여 데이터 분포를 확대하고 판별자는 convolutional layers을 사용하여 데이터를 분류한다. 

>#### Super-resolution GAN

Super-resolution GAN(SRGAN)은 저해상도 이미지를 고해상도로 상향하는 데 중점을 둔다. 목표는 이미지 품질과 세부 사항을 유지하면서 이미지를 더 높은 해상도로 상향하는 것이다.

<br>

이 외에도 StyleGAN, CycleGAN, DiscoGAN 등 다양한 유형의 문제를 해결하는 다른 GAN 유형들이 있다.

<br>

### GAN의 장단점 
GAN은 매우 강력한 생성 모델이지만, 몇 가지 단점도 가지고 있다.

>#### 장점
- 고품질의 이미지 생성: GAN은 매우 정교하고 현실적인 이미지를 생성할 수 있다
- 다양한 응용 가능성: 이미지 생성, 데이터 증강, 스타일 변환 등 다양한 분야에 활용할 수 있다

>#### 단점
- **학습 불안정성**: GAN의 학습은 매우 불안정할 수 있으며, 적절한 하이퍼파라미터 설정이 필요하다
- **모드 붕괴**: 생성자가 특정 유형의 이미지만 생성하고 다양한 이미지를 생성하지 못하는 문제가 발생할 수 있다
- **손실값 모니터링의 어려움**: GAN의 손실 함수를 모니터링하기 어려워 학습 과정을 추적하는 데 어려움이 있다

<br><br>

## 3. 딥페이크 탐지 시스템의 작동 원리

1. 이미지/픽셀 기반 탐지
   - CNN(Convolutional Neural Networks)과 같은 딥러닝 모델을 사용하여 이미지 내 조작 흔적을 찾음
   - 얼굴 영역의 불일치, 색상/텍스처 패턴의 비정상적 특징 등을 감지
   - EfficientNet, ResNet 등의 아키텍처가 자주 사용된다

2. 신체특징 기반 탐지
   - 눈 깜빡임, 얼굴 표정, 맥박신호(rPPG), 눈/입 동기화 등 생체적 특징을 분석

3. 주파수 기반 탐지
   - FFT(Fast Fourier Transform)와 같은 주파수 도메인 변환을 통해 분석
   - GAN 생성 이미지는 특정 주파수 패턴을 가지는 경향이 있어, 이를 통해 탐지
   - DCT(Discrete Cosine Transform) 계수 분석도 활용

4. 시간적 일관성 분석
   - 비디오에서 프레임 간 일관성을 분석하여 딥페이크 탐지
   - 시간에 따른 얼굴 특징의 변화가 자연스러운지 확인
   - 3D 얼굴 모델링을 통한 일관성 분석도 포함된다

5. 메타데이터/압축 아티팩트 분석
   - 이미지/비디오 메타데이터 및 압축 특성을 분석
   - 이중 압축 흔적, 노이즈 패턴 등을 활용
  
<br><br>


## 4. 딥페이크 영상 탐지 AI에 필요한 것

### 개발 환경과 프레임워크 

- 딥러닝 프레임워크 : PyTorch, TensorFlow, Keras
- 이미지 및 비디오 처리 : OpenCV
- 머신러닝 알고리즘 및 평가도구 : scikit-learn
- 얼굴 감지 및 랜드마크 추출 : Dlib
- 데이터 처리 및 분석 : Numpy, Pandas

<br>

### 데이터셋

- [FaceForensics++](https://github.com/ondyari/FaceForensics) : 다양한 딥페이크 기법으로 조작된 비디오 데이터셋
- [DeepFake Detection Challenge (DFDC)](https://ai.meta.com/datasets/dfdc/) : Facebook 주관의 대규모 데이터셋
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) : 유명 인물 기반 딥페이크 데이터셋
- DF-TIMIT : 음성과 영상 모두 포함된 딥페이크 데이터셋
- UADFV (UCLA Anomaly Detection Dataset for Video ): 비디오 기반 이상 탐지 데이터셋

<br>

### 모델 아키텍쳐

- [ResNet (Residual Network)](https://arxiv.org/abs/1512.03385) : 이미지 분류 및 특징 추출
- [EfficientNet](https://arxiv.org/abs/1905.11946) : 고효율 CNN 모델
- [XceptionNet](https://arxiv.org/abs/1610.02357) : 딥페이크 탐지에 효과적
- Vision Transformer (ViT) : 이미지 분류 및 이상 탐지
- LSTM (Long Short-Term Memory) : 비디오 시퀀스 분석



<br><br>

## 5. 관련 문서

- What is GAN? : https://aws.amazon.com/what-is/gan/?nc1=h_ls
- GAN에 대한 이해 : https://medium.com/@hugmanskj/gan%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4-a073a5425ef2
- Autoencoders in Machine Learning : https://www.geeksforgeeks.org/auto-encoders/
- What is an Autoencoder : https://www.ibm.com/think/topics/autoencoder
- 누가 진짜일까? 딥페이크 생성과 탐지 : https://www.samsungsds.com/kr/insights/220411deepfake_1.html
- Keras vs Tensorflow vs Pytorch : https://www.geeksforgeeks.org/keras-vs-tensorflow-vs-pytorch/
- 딥페이크 데이터셋/툴/논문/코드 모음 : https://github.com/Daisy-Zhang/Awesome-Deepfakes?tab=readme-ov-file
- Generative Models in AI: A Comprehensive Comparison of GANs and VAEs : https://www.geeksforgeeks.org/generative-models-in-ai-a-comprehensive-comparison-of-gans-and-vaes/

<br>

그 외 : 제 깃허브의 star list 참고 부탁드립니다 

<br><br>

---
## 6. 관련 용어 정리 (추가예정)
