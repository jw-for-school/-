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
현재는 딥페이크 기술에 대한 접근성이 용이해져 상업적 발전을 이뤄 인더스트리에서 기술이 발전되고 있다

이처럼 딥페이크는 인공지능 기술을 발전에 힘입어 상업적으로 많은 발전을 이루며 일반인들에게도 접근성이 낮아졌지만,  ~와 같은 사회적 문제 또한 심화되고 있다

//딥페이스 기술을 나타내는 이미지1, 늘어나는 딥페이크 범죄 표를 나타낸 이미지2


<br><br>


## 2. 딥페이크의 작동 원리 

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
- 누가 진짜일까? 딥페이크 생성과 탐지 : https://www.samsungsds.com/kr/insights/220411deepfake_1.html
- Keras vs Tensorflow vs Pytorch : https://www.geeksforgeeks.org/keras-vs-tensorflow-vs-pytorch/
- Dlib : https://github.com/davisking/dlib
- 딥페이크 데이터셋/툴/논문/코드 모음 : https://github.com/Daisy-Zhang/Awesome-Deepfakes?tab=readme-ov-file
- 오토인코더란?
- GAN vs Autoencoder

<br>

그 외 : 제 깃허브의 star list 참고 부탁드립니다 

<br><br>

---
## 6. 관련 용어 정리 (추가예정)
