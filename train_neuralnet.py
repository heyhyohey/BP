# 1. 모듈 임포트
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 2. 데이터 읽기(mnist 데이터 셋, x_train, t_train 학습 데이터, x_test, t_test 검증 데이터)
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# 3. 입력층 784, 은닉층 50, 출력층 10의 모델 생성
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 4. 학습에 필요한 수치 정의
# 4-1. 반복횟수
iters_num = 10000
# 4-2. 학습 데이터의 크기
train_size = x_train.shape[0]
# 4-3. 학습 데이터의 배치 단위의 크기(100개씩 나눠서 학습 수행)
batch_size = 100
# 4-4. 학습률
learning_rate = 0.1
# 4-5. 손실함수, 정확도 값을 저장할 리스트 정의
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 4-6. 학습 데이터의 크기를 배치의 크
iter_per_epoch = max(train_size / batch_size, 1)

# 5. 모델 생성
for i in range(iters_num):
    # 5-1. 랜덤으로 배치를 선택
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 5-2. 오차역전파법으로 기울기를 구함
    grad = network.gradient(x_batch, t_batch)

    # 5-3. 가중치 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 5-4. 손실함수 값을 구함
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 5-5. 정확도 출력
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
