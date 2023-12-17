'''
注意 Field
https://github.com/pytorch/text/issues/1274
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.legacy.data import Field, LabelField, TabularDataset, BucketIterator
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 定义字段
text_field = Field(sequential=True, lower=True, batch_first=True)
label_field = LabelField(dtype=torch.float)

# 加载IMDB电影评论数据集
train_data, test_data = IMDB.splits(text_field, label_field)

# 构建词汇表
text_field.build_vocab(train_data, max_size=5000)

# 创建数据迭代器
batch_size = 64
train_iter, test_iter = BucketIterator.splits(
    (train_data, test_data),
    batch_sizes=(batch_size, batch_size),
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.transformer_encoder(embedded)
        pooled = encoded.mean(dim=1)
        logits = self.fc(pooled)
        return logits.squeeze(1)

# 设置模型参数
vocab_size = len(text_field.vocab)
d_model = 64
nhead = 4
num_layers = 2

# 初始化模型
model = TransformerModel(vocab_size, d_model, nhead, num_layers)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for batch in train_iter:
        optimizer.zero_grad()
        logits = model(batch.text)
        loss = criterion(logits, batch.label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# 在测试集上评估模型
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for batch in test_iter:
        logits = model(batch.text)
        predicted_labels = (logits > 0).long()
        total_correct += (predicted_labels == batch.label.long()).sum().item()
        total_samples += batch.label.size(0)
accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy:.4f}")