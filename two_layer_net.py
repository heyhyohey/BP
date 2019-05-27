# 1. 모듈 임포트
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 1. 가중치 초기화
        self.params = {}
        # 1-1. 1번째 계층의 weight(가중치) 초기화
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        # 1-2. 1번째 계층의 bias(편향) 초기화
        self.params['b1'] = np.zeros(hidden_size)
        # 1-3. 2번째 계층의 weight(가중치) 초기화
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        # 1-4. 2번째 계층의 bias(편향) 초기화
        self.params['b2'] = np.zeros(output_size)

        # 2. 계층 생성
        self.layers = OrderedDict()
        # 2-1. 1번째 Affine 계층 생성
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        # 2-2. ReLU 계층 생성
        self.layers['Relu1'] = Relu()
        # 2-3. 2번째 Affine 계층 생성
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])
        # 2-4. Softmax 계층 생성(손실함수 포함)
        self.lastLayer = SoftmaxWithLoss()

    # 추론을 수행하는 함수(x는 이미지 데이터)
    def predict(self, x):
        for layer in self.layers.values():
            # 계층에 데이터를 넣고 순전파
            x = layer.forward(x)
        # 순전파 후의 결과를 리턴
        return x

    # 손실 함수의 값을 구하는 함수(x : 입력 데이터, t : 정답 레이블)
    def loss(self, x, t):
        # 1. 값을 추론함
        y = self.predict(x)
        # 2. 추론된 값을 바탕으로 손실 함수의 값을 구하여 반환
        return self.lastLayer.forward(y, t)

    # 정확도를 구하는 함수
    def accuracy(self, x, t):
        # 1. 값을 추론함
        y = self.predict(x)
        # 2. 추론된 값 중에서 가장 큰 값을 받음
        y = np.argmax(y, axis=1)
        # 3. 타겟 데이터가 1개가 아니라면 가장 큰 값을 골라냄
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        # 4. 정확도는 예측된 값과 정답 값이 같을 경우의 합을 전체 데이터의 양으로 나눈 값
        accuracy = np.sum(y == t) / float(x.shape[0])

        # 5. 정확도를 반환
        return accuracy

    #  가중치 매개변수의 기울기를 미분 방식으로 구함(x : 입력 데이터, t : 정답 레이블)
    def numerical_gradient(self, x, t):
        # 1. 손실함수의 값을 구함
        loss_W = lambda W: self.loss(x, t)

        # 2. 가중치 매개변수의 기울기를 미분 방식으로 구함
        grads = {}
        # 2-1. 1번째 층의 가중치의 기울기를 구함
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 2-1. 1번째 층의 편향의 기울기를 구함
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 2-1. 2번째 층의 가중치의 기울기를 구함
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 2-1. 2번째 층의 편향의 기울기를 구함
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        # 3. 기울기의 변화량을 반환
        return grads

    # 가중치 매개변수의 기울기를 오차역전파법으로 구함
    def gradient(self, x, t):
        # 1. 순전파
        self.loss(x, t)

        # 2. 역전파
        dout = 1

        # 2-1. 마지막 층 부터 역전파를 수행하여 시작
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        # 2-2. 전체 층에 대해서 역전파를 수행
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 3. 결과 저장
        grads = {}
        # 3-1. 1번째 Affine 계층의 가중치 변화량 저장
        grads['W1'] = self.layers['Affine1'].dW
        # 3-2. 1번째 Affine 계층의 편향 변화량 저장
        grads['b1'] = self.layers['Affine1'].db
        # 3-3. 2번째 Affine 계층의 가중치 변화량 저장
        grads['W2'] = self.layers['Affine2'].dW
        # 3-4. 2번째 Affine 계층의 편향 변화량 저장
        grads['b2'] = self.layers['Affine2'].db

        # 4. 기울기 변화량을 반환
        return grads
