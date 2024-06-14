train_path = '../C6/train.txt'
test_path = '../C6/test.txt'
valid_path = '../C6/valid.txt'
word2vec_path = '../C7/GoogleNews-vectors-negative300.bin'
checkpoint_path = 'model.pt'

# A70 単語ベクトルの和による特徴量
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors


# word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def A70():
    train_data = pd.read_csv(train_path, sep='\t', header=None)
    test_data = pd.read_csv(test_path, sep='\t', header=None)
    valid_data = pd.read_csv(valid_path, sep='\t', header=None)
    path_list = ['train_vectorized_data.csv', 'train_label_data.csv', 'test_vectorized_data.csv', 'test_label_data.csv',
                 'valid_vectorized_data.csv', 'valid_label_data.csv']
    label_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    for data in [train_data, test_data, valid_data]:
        title_data = []
        label_data = []

        for i, row in data.iterrows():
            title_vec = np.zeros(300)
            # TODO
            # If KeyError, len should be less
            for word in row[1].split(' '):
                try:
                    title_vec += word2vec[word]
                except KeyError:
                    pass
            label_data.append(label_dict[row[0]])
            title_data.append(title_vec / len(row[1].split(' ')))

        title_data = pd.DataFrame(title_data)
        label_data = pd.DataFrame(label_data)
        title_data.to_csv(path_list.pop(0), index=False)
        label_data.to_csv(path_list.pop(0), index=False)


import torch

train_data = pd.read_csv('train_vectorized_data.csv')
# test_data = pd.read_csv('test_vectorized_data.csv')
valid_data = pd.read_csv('valid_vectorized_data.csv')
train_label_data = pd.read_csv('train_label_data.csv', sep='\t', index_col=False)
# test_label_data = pd.read_csv('test_label_data.csv', sep='\t', index_col=False)
valid_label_data = pd.read_csv('valid_label_data.csv', sep='\t', index_col=False)

w = torch.randn(300, 4, dtype=torch.float64, requires_grad=True)
sftmax = torch.nn.Softmax()

# A71 単層ニューラルネットワークによる予測
x_1 = torch.tensor(train_data.iloc[0].values, requires_grad=True)
x_1_4 = torch.tensor(train_data.iloc[0:4].values, requires_grad=True)


def A71() -> tuple:
    y_hat_1 = sftmax(x_1 @ w)
    y_hat = sftmax(x_1_4 @ w)

    return y_hat_1[None], y_hat


# A72 損失と勾配の計算
nll_loss = torch.nn.NLLLoss()


def A72() -> torch.Tensor:
    y_hat_1, y_hat = A71()
    loss_1 = nll_loss(y_hat_1, torch.tensor(train_label_data.iloc[0].values, dtype=torch.long))
    sftmax.zero_grad()
    loss_1.backward()
    print(f'loss_1: {loss_1}')
    print(f'w.grad: {w.grad}')

    loss_2 = nll_loss(y_hat, torch.squeeze(torch.tensor(train_label_data.iloc[0:4].values, dtype=torch.long)))
    sftmax.zero_grad()
    loss_2.backward()
    print(f'loss_2: {loss_2}')
    print(f'w.grad: {w.grad}')


# A73 確率的勾配降下法による学習
# A75 損失と正解率のプロット
# A76 チェックポイント
# A77 ミニバッチ化
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time

optimizer = optim.SGD([w], lr=0.01)


def A73(epoch: int, batch_size: int = 1):
    checkpoints = []

    x_plot = []
    y_loss_plot = []
    y_acc_plot = []
    y_valid_loss_plot = []
    y_valid_acc_plot = []

    dataset = TensorDataset(torch.tensor(train_data.values, requires_grad=True),
                            torch.squeeze(torch.tensor(train_label_data.values, dtype=torch.long)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    sftmax.to(device)

    for i in range(epoch):

        start_time = time.time()
        for _, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            y_hat = sftmax(X @ w)
            loss = nll_loss(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        print(f'total time for batch size {batch_size} at epoch {i}: {end_time - start_time}')
        break

        # print(f'epoch: {i}, loss: {loss}')

        checkpoints.append({
            'epoch': i,
            'w': w,
            'model_state_dict': sftmax.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        })

        if i % 20 == 0:
            train_acc, valid_acc = A74()
            x_valid = torch.tensor(valid_data.values, requires_grad=True)
            loss_valid = nll_loss(sftmax(x_valid @ w),
                                  torch.squeeze(torch.tensor(valid_label_data.values, dtype=torch.long)))

            # print(f'train_accuracy: {train_acc}, valid_accuracy: {valid_acc} at time step: {i}')

            x_plot.append(i)
            y_loss_plot.append(loss.item())
            y_acc_plot.append(train_acc)
            y_valid_loss_plot.append(loss_valid.item())
            y_valid_acc_plot.append(valid_acc)

            plt.figure()

            ax1 = plt.subplot(221)
            ax1.plot(x_plot, y_acc_plot, label='accuracy', color='red', marker='o')
            ax1.plot(x_plot, y_loss_plot, label='loss', color='blue', marker='o')
            ax1.set_title('train')
            ax1.legend()
            ax2 = plt.subplot(222)
            ax2.plot(x_plot, y_valid_acc_plot, label='accuracy', color='red', marker='o')
            ax2.plot(x_plot, y_valid_loss_plot, label='loss', color='blue', marker='o')
            ax2.set_title('valid')
            ax2.legend()

            plt.show()

    torch.save(checkpoints, checkpoint_path)


# A74 正解率の計測
def A74():
    train_x = torch.tensor(train_data.values, requires_grad=True)
    train_y = torch.squeeze(torch.tensor(train_label_data.values, dtype=torch.long))
    train_y_hat = sftmax(train_x @ w)
    train_accuracy = 0

    for _y, _y_hat in zip(train_y, train_y_hat):
        if _y == torch.argmax(_y_hat):
            train_accuracy += 1

    valid_x = torch.tensor(valid_data.values, requires_grad=True)
    valid_y = torch.squeeze(torch.tensor(valid_label_data.values, dtype=torch.long))
    valid_y_hat = sftmax(valid_x @ w)
    valid_accuracy = 0

    for _y, _y_hat in zip(valid_y, valid_y_hat):
        if _y == torch.argmax(_y_hat):
            valid_accuracy += 1

    return (train_accuracy / len(train_y), valid_accuracy / len(valid_y))


def test_chapter8():
    # A70()
    # print(A71())
    # print(A72())
    A73(200, 16)
    # print(A74())


if __name__ == '__main__':
    test_chapter8()
