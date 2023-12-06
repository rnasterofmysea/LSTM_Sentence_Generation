# LSTM_Sentence_Generation

![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/136e6b8b-6a65-46c7-8854-e00866042513)


![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/70c43b40-6d11-400a-bed3-05bc8e1a545a)


## 시스템 구조
![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/9d11ea30-dbff-48f6-bf3b-5fc2925f1efd)


### (pre) 데이터 살펴보기

```
import pandas as pd
import os
import string

df = pd.read_csv("ArticlesApril2017.csv")
print(df.columns)
```
![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/399a4b6d-2faa-45f3-adfd-36167c256b3f)

![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/58a91fa8-c3c1-46ed-9678-797bf0ada445)

### 학습용 데이터셋 정의 - dataset_generation.py

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
        for filename in glob.glob("ArticlesApril2018.csv"):
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

### LSTM모델 정의 - model.py

```
import torch
import torch.nn as nn


class LSTM(nn.Module):
   def __init__(self, num_embeddings):
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

### 유틸리티 - utill.py

```

def calculate_accuracy(preds, y):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(correct)
    return acc


def data_split(dataset,args):
  # 데이터셋 분할 비율 설정
  train_ratio = args.train_ratio
  val_ratio = args.val_ratio
  test_ratio = 1 - train_ratio - val_ratio
  total_epoch =2
  # 데이터셋 분할
  total_size = len(dataset)
  train_size = int(train_ratio * total_size)
  val_size = int(val_ratio * total_size)
  test_size = total_size - train_size - val_size

  train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

  # DataLoader 설정
  train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=arg.batch_size)
  test_loader = DataLoader(test_dataset, batch_size=arg.batch_size)
  data_loaders = [train_loader, val_loader, test_loader]
  return  data_loaders
```

### 모델 학습하기 --train.py

```
import torch
import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


# 학습을 진행할 프로세서 정의
device = "cuda" if torch.cuda.is_available() else "cpu"
# 데이터셋 및 모델 설정
dataset = TextGeneration()  # 가정된 데이터셋
model = LSTM(num_embeddings=len(dataset.BOW)).to(device)  # LSTM 모델
optim = Adam(model.parameters(), lr=0.001)  # 옵티마이저
loss_function = CrossEntropyLoss()  # 손실 함수

# 데이터셋 분할 비율 설정
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
total_epoch =50

# 데이터셋 분할
total_size = len(dataset)
train_size = int(train_ratio * total_size)
val_size = int(val_ratio * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

#기록 리스트들
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []


# 학습 루프
for epoch in range(total_epoch):
    model.train()
    train_loss = 0
    train_acc = 0
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        # 기울기 초기화
        optim.zero_grad()

        # 데이터를 device에 로드
        data = torch.tensor(data, dtype=torch.long).to(device)
        label = torch.tensor(label, dtype=torch.long).to(device)

        # 모델의 예측
        pred = model(data)

        # 손실 계산
        loss = loss_function(pred, label)
        acc = calculate_accuracy(pred, label)

        # 역전파 및 최적화
        loss.backward()
        optim.step()
        train_loss += loss.item()
        train_acc += acc.item()

        iterator.set_description(f"epoch{epoch} loss:{loss.item()}")

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)

    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc)
    tqdm.tqdm.write(f'Epoch {epoch}: 평균 학습 손실 {avg_train_loss}')

    # 검증 루프
    model.eval()
    val_loss = 0
    valid_acc = 0
    with torch.no_grad():
        for data, label in val_loader:
            data = torch.tensor(data, dtype=torch.long).to(device)
            label = torch.tensor(label, dtype=torch.long).to(device)
            pred = model(data)
            loss = loss_function(pred, label)
            acc = calculate_accuracy(pred, label)

            val_loss += loss.item()
            valid_acc += acc.item()


    avg_val_loss = val_loss / len(val_loader)
    avg_valid_acc = valid_acc / len(val_loader)
    
    valid_losses.append(avg_val_loss)
    valid_accuracies.append(avg_valid_acc)    
    tqdm.tqdm.write(f'Epoch {epoch}: 평균 검증 손실 {avg_val_loss}')



### 모델 저장
torch.save(model.state_dict(), "lstm.pth")
```

![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/a0cfa521-dff9-4d92-9b6a-cd976be4f23a)

### 학습 결과 시각화

```
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 준비
data = {
    "Epoch": range(1, total_epoch + 1),
    "Train Loss": train_losses,
    "Validation Loss": valid_losses,
    "Train Accuracy": train_accuracies,
    "Validation Accuracy": valid_accuracies
}

# DataFrame 생성
df = pd.DataFrame(data)

# DataFrame 출력
print(df)

# 시각화
plt.figure(figsize=(12, 6))

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss")
plt.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(df["Epoch"], df["Train Accuracy"], label="Train Accuracy")
plt.plot(df["Epoch"], df["Validation Accuracy"], label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

```
![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/fee73d14-e41c-493c-b8ee-1b127fbc8a85)


### 테스트 _ test.py

```
# 테스트 루프
model.eval()
test_loss = 0
with torch.no_grad():
    for data, label in test_loader:
        data = torch.tensor(data, dtype=torch.long).to(device)
        label = torch.tensor(label, dtype=torch.long).to(device)

        pred = model(data)
        loss = loss_function(pred, label)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f'테스트 손실: {avg_test_loss}')
```

![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/30d2753d-0a9a-42c2-b3c7-8007510d6b07)


### 문장 생성하기 predict.py

```
def generate(model, BOW, string="finding my ", strlen=2):
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
![image](https://github.com/rnasterofmysea/LSTM_Sentence_Generation/assets/81907470/48c58c2d-9a28-4678-8357-d65b3f240850)
