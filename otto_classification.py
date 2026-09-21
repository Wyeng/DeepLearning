import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# 数据准备
train_data=pd.read_csv('../data/otto/train.csv')
test_data=pd.read_csv('../data/otto/test.csv')
x_train=train_data.iloc[:,1:94].values
y_train=train_data.target
x_test=test_data.iloc[:,1:94].values

# 数据预处理
label_encoder=LabelEncoder()
y_train=label_encoder.fit_transform(y_train)

# 将训练集和测试集转化为tensor类型
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)


# 创建训练集和测试集的dataset和dataloader
train_dataset=TensorDataset(x_train_tensor,y_train_tensor)
test_dataset=TensorDataset(x_test_tensor)
train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=32)

# 构建模型
class model(torch.nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.l1=torch.nn.Linear(93,128)
        self.l2=torch.nn.Linear(128,64)
        self.l3=torch.nn.Linear(64,9)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        return self.l3(x)

model=model()
model=model.cuda()
# 损失函数
loss_fn=torch.nn.CrossEntropyLoss()
loss_fn=loss_fn.cuda()
# 优化器
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

def trian(train_epoch):
    for epoch in range(train_epoch):
        total_loss=0
        for x_train,y_train in train_dataloader:
            x_train=x_train.cuda()
            y_train=y_train.cuda()
            optimizer.zero_grad()
            y_pred=model(x_train)
            loss=loss_fn(y_pred,y_train)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        if epoch%10==0:
            print(f'epoch:{epoch},loss:{total_loss}')
        if epoch==train_epoch-1:
            torch.save(model.state_dict(),'../data/otto/model.pt')

def test():
    model.load_state_dict(torch.load('../data/otto/model.pt'))
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for x_test_batch in test_dataloader:
            x_test = x_test_batch[0].cuda()
            y_pred = model(x_test)
            # 使用softmax获取概率分布
            y_pred = F.softmax(y_pred, dim=1)
            y_pred = y_pred.cpu().numpy()
            all_predictions.append(y_pred)

    # 合并所有批次的预测结果
    all_predictions = np.vstack(all_predictions)

    # 将每行最大概率位置设为1，其余设为0（整数类型）
    result_matrix = np.zeros_like(all_predictions, dtype=int)
    max_indices = np.argmax(all_predictions, axis=1)
    result_matrix[np.arange(len(max_indices)), max_indices] = 1

    # 读取测试数据的ID
    test_data = pd.read_csv('../data/otto/test.csv')
    ids = test_data['id'].values

    # 创建结果DataFrame
    columns = [f'Class_{i}' for i in range(1, 10)]
    result_df = pd.DataFrame(result_matrix, columns=columns)
    result_df.insert(0, 'id', ids)

    # 保存为CSV文件
    result_df.to_csv('../data/otto/submission.csv', index=False)

if __name__ == '__main__':
    # train_epoch=100
    # trian(train_epoch)
    test()