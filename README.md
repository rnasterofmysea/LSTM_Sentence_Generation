# LSTM_Sentence_Generation

## (pre) 데이터 살펴보기

```
import pandas as pd
import os
import string

df = pd.read_csv("ArticlesApril2017.csv")
print(df.columns)
```
![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/399a4b6d-2faa-45f3-adfd-36167c256b3f)

![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/58a91fa8-c3c1-46ed-9678-797bf0ada445)


## model.py -- LSTM 모델 정의

```
import torch
import torch.nn as nn


class LSTM(nn.Module):
   def __init__(self, num_embeddings, input_size, hidden_size, num_layers, batch_first):
       super(LSTM, self).__init__()

       # ❶ 밀집표현을 위한 임베딩층
       self.embed = nn.Embedding(
           num_embeddings=num_embeddings, embedding_dim=16)

       # LSTM을 5개층을 쌓음
       self.lstm = nn.LSTM(
           input_size=16,
           hidden_size=64,
           num_layers=5,
           batch_first=True)

       # 분류를 위한 MLP층
       self.fc1 = nn.Linear(128, num_embeddings)
       self.fc2 = nn.Linear(num_embeddings,num_embeddings)

       # 활성화 함수
       self.relu = nn.ReLU()

   def forward(self, x):
       x = self.embed(x)

       # ❷ LSTM 모델의 예측값
       x, _ = self.lstm(x)
       x = torch.reshape(x, (x.shape[0], -1))
       x = self.fc1(x)
       x = self.relu(x)
       x = self.fc2(x)

       return x
```

## text_preprocessing.py -- 단어 전처리

```
import numpy as np
import glob

from torch.utils.data.dataset import Dataset


class TextGeneration(Dataset):
    def clean_text(self, txt):
        # 모든 단어를 소문자로 바꾸고 특수문자를 제거
        txt = "".join(v for v in txt if v not in string.punctuation).lower()
        return txt
    def __init__(self):
        all_headlines = []

        # ❶ 모든 헤드라인의 텍스트를 불러옴
        for filename in glob.glob("ArticlesApril2017.csv"):
            if 'Articles' in filename:
                article_df = pd.read_csv(filename)

                # 데이터셋의 headline의 값을 all_headlines에 추가
                all_headlines.extend(list(article_df.headline.values))
                break

        # ❷ headline 중 unknown 값은 제거
        all_headlines = [h for h in all_headlines if h != "Unknown"]

        # ❸ 구두점 제거 및 전처리가 된 문장들을 리스트로 반환
        self.corpus = [self.clean_text(x) for x in all_headlines]
        self.BOW = {}

        # ➍ 모든 문장의 단어를 추출해 고유번호 지정
        for line in self.corpus:
            for word in line.split():
                if word not in self.BOW.keys():
                    self.BOW[word] = len(self.BOW.keys())

        # 모델의 입력으로 사용할 데이터
        self.data = self.generate_sequence(self.corpus)
    def generate_sequence(self, txt):
        seq = []

        for line in txt:
            line = line.split()
            line_bow = [self.BOW[word] for word in line]

            # 단어 2개를 입력으로, 그다음 단어를 정답으로
            data = [([line_bow[i], line_bow[i+1]], line_bow[i+2])
            for i in range(len(line_bow)-2)]

            seq.extend(data)

        return seq
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        data = np.array(self.data[i][0])  # ❶ 입력 데이터
        label = np.array(self.data[i][1]).astype(np.float32)  # ❷ 출력 데이터

        return data, label
```

## utill.py --학습용 데이터셋 정의

```
class CustomTextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = np.array(self.data[i][0])
        label = np.array(self.data[i][1]).astype(np.float32)
        return data, label

def AllocateDataset(full_data, train_size, valid_size, test_size):

  # Splitting the data into training and the rest (validation + test)
  train_data, rest_data = train_test_split(full_data, test_size=0.3, random_state=42)

  # Splitting the rest into validation and test
  valid_data, test_data = train_test_split(rest_data, test_size=0.5, random_state=42)
  train_dataset = CustomTextDataset(train_data)
  valid_dataset = CustomTextDataset(valid_data)
  test_dataset = CustomTextDataset(test_data)

  dataset = [train_dataset,valid_dataset,test_dataset]
  return dataset
```

## train.py -- 모델 학습하기

```
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam

# 학습을 진행할 프로세서 정의
device = "cuda" if torch.cuda.is_available() else "cpu"

train_size = 0.6
valid_size = 0.2
test_size = 0.2
total_epoch = 10
input_size=16,
hidden_size=64,
num_layers=5
batch_first=True

dataset = TextGeneration()  # 데이터셋 정의


# Creating datasets for training, validation, and testing
data_split = AllocateDataset(dataset.data, train_size, valid_size, test_size)

# Data loaders for each dataset
train_loader = DataLoader(data_split[0], batch_size=64, shuffle=True)
valid_loader = DataLoader(data_split[1], batch_size=64, shuffle=True)
test_loader = DataLoader(data_split[2], batch_size=64, shuffle=True)


model = LSTM(num_embeddings=len(dataset.BOW),input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first).to(device)  # 모델 정의
optim = Adam(model.parameters(), lr=0.001)

train_losses = []  # List to store training loss values
valid_losses = []  # List to store validation loss values

for epoch in range(total_epoch):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        # 기울기 초기화
        optim.zero_grad()

        # 모델의 예측값
        pred = model(torch.tensor(data, dtype=torch.long).to(device))

        # 정답 레이블은 long 텐서로 반환해야 함
        loss = nn.CrossEntropyLoss()(pred, torch.tensor(label, dtype=torch.long).to(device))

        # 오차 역전파
        loss.backward()
        optim.step()
        train_losses.append(loss.item())
        iterator.set_description(f"epoch{epoch} loss:{loss.item()}")

    # Validation Phase
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for data, label in valid_loader:
            pred = model(torch.tensor(data, dtype=torch.long).to(device))
            loss = nn.CrossEntropyLoss()(pred, torch.tensor(label, dtype=torch.long).to(device))
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)  # Storing the average validation loss
    print(f"Epoch {epoch}: Validation Loss = {avg_valid_loss}")

torch.save(model.state_dict(), "lstm.pth")
```

## 학습 결과 분석

```
import matplotlib.pyplot as plt

# Plotting training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
# plt.plot(valid_losses, label='Validation Loss')
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```
![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/24218ba8-6340-4bb1-8db8-0ef5bfe81009)

## test.py --모델 테스트

```
test_loader = DataLoader(data_split[2], batch_size=64, shuffle=True)
test_loss = 0
model.eval()
with torch.no_grad():
    for data, label in test_loader:
        pred = model(torch.tensor(data, dtype=torch.long).to(device))
        loss = nn.CrossEntropyLoss()(pred, torch.tensor(label, dtype=torch.long).to(device))
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss = {avg_test_loss}")
```

## predict.py --문장 생성하기

```
def generate(model, BOW, string="finding an ", strlen=10):
   device = "cuda" if torch.cuda.is_available() else "cpu"

   print(f"input word: {string}")

   with torch.no_grad():
       for p in range(strlen):
           # 입력 문장을 텐서로 변경
           words = torch.tensor(
               [BOW[w] for w in string.split()], dtype=torch.long).to(device)

           # ❶
           input_tensor = torch.unsqueeze(words[-2:], dim=0)
           output = model(input_tensor)  # 모델을 이용해 예측
           output_word = (torch.argmax(output).cpu().numpy())
           string += list(BOW.keys())[output_word]  # 문장에 예측된 단어를 추가
           string += " "

   print(f"predicted sentence: {string}")

model.load_state_dict(torch.load("lstm.pth", map_location=device))
pred = generate(model, dataset.BOW)
```

![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/1fcec743-f380-4e5e-9576-cca2a8f71c2d)
