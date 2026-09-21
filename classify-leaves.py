import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import torchvision
from torchvision.models import ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用预训练模型，并进行微调
pretrained_model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
pretrained_model.fc = nn.Linear(in_features=512, out_features=176)
model = pretrained_model.to(device)
nn.init.xavier_uniform_(model.fc.weight)


# 数据集
class CSVLabelDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # 如果是类别名称，自动编码
        if self.data['label'].dtype == 'object':
            self.classes = self.data['label'].unique()
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.data['label'] = self.data['label'].map(self.class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = CSVLabelDataset(
    csv_file='../data/classify-leaves/train.csv',
    root_dir='../data/classify-leaves',
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)


class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name  # 返回图像和文件名，用于后续保存结果


# 注意：测试集也需要做与训练集相同的归一化
test_dataset = TestDataset("../data/classify-leaves/test.csv",
                           "../data/classify-leaves",
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                           ]))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 损失函数
criterion = nn.CrossEntropyLoss()


def train(epochs):
    best_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0.0  # 记录整个epoch的累计损失
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 100 == 0:
                print('Epoch:{}, batch:{}, Loss:{}'.format(epoch, i, loss.item()))

        # 计算平均损失并保存最佳模型
        avg_loss = epoch_loss / len(train_dataloader)
        print('Epoch:{}, Average Loss:{}'.format(epoch, avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), '../model/best_model.pth')
            print('Best model saved with loss:', best_loss)


def evaluate(output_file='../data/classify-leaves/sample_submission.csv'):
    # 加载训练好的模型权重
    model.load_state_dict(torch.load('../model/best_model.pth', map_location=device))
    model.eval()

    # 获取类别映射（从训练数据集中获取）
    train_data = pd.read_csv('../data/classify-leaves/train.csv')

    # 构建类别映射
    if train_data['label'].dtype == 'object':
        classes = train_data['label'].unique()
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    else:
        # 如果label已经是数字，需要从其他来源获取类别名称
        # 这里假设类别名称就是数字对应的字符串，或者你可以手动定义
        idx_to_class = {i: str(i) for i in range(176)}  # 176个类别

    predictions = []
    image_names = []

    with torch.no_grad():
        for inputs, names in test_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # 将预测结果转换为类别名称
            predicted_labels = [idx_to_class[idx.item()] for idx in predicted]
            predictions.extend(predicted_labels)
            image_names.extend(names)

    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'image': image_names,
        'label': predictions
    })

    # 保存到指定的sample_submission.csv文件
    result_df.to_csv(output_file, index=False)
    print(f"预测完成，结果已保存到 {output_file}，共 {len(predictions)} 条记录")

    return result_df


if __name__ == '__main__':
    # 训练模型（如果需要）
    # train(epochs=10)

    # 评估并生成提交文件
    evaluate(output_file='../data/classify-leaves/sample_submission.csv')