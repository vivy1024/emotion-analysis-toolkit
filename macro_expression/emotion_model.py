#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FER2013表情识别模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 定义表情标签
EMOTIONS = {
    0: "生气",     # Angry
    1: "厌恶",     # Disgust
    2: "恐惧",     # Fear
    3: "开心",     # Happy
    4: "难过",     # Sad
    5: "惊讶",     # Surprise
    6: "中性"      # Neutral
}

class EmotionCNN(nn.Module):
    """
    基础CNN模型用于情感识别
    
    输入：灰度图像 (1, 48, 48)
    输出：7种情绪的概率分布
    """
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第四个卷积块
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 7)  # 7种情绪类别
    
    def forward(self, x):
        # 卷积块1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积块2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 卷积块3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 卷积块4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

class DeepEmotionCNN(nn.Module):
    """
    更深层的CNN模型用于情感识别
    
    输入：灰度图像 (1, 48, 48)
    输出：7种情绪的概率分布
    """
    def __init__(self):
        super(DeepEmotionCNN, self).__init__()
        
        # 第一个卷积块
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积块
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第四个卷积块
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 7)  # 7种情绪类别
    
    def forward(self, x):
        # 卷积块1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.pool1(F.relu(self.bn1_2(self.conv1_2(x))))
        
        # 卷积块2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.pool2(F.relu(self.bn2_2(self.conv2_2(x))))
        
        # 卷积块3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = self.pool3(F.relu(self.bn3_2(self.conv3_2(x))))
        
        # 卷积块4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = self.pool4(F.relu(self.bn4_2(self.conv4_2(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class EmotionResNet(nn.Module):
    """
    基于ResNet的情感识别模型
    
    使用预训练的ResNet18作为特征提取器
    输入：灰度图像 (1, 48, 48)
    输出：7种情绪的概率分布
    """
    def __init__(self, num_classes=7):
        super(EmotionResNet, self).__init__()
        
        # 加载预训练的ResNet18模型
        self.resnet = models.resnet18(weights='DEFAULT')
        
        # 修改第一个卷积层以接受灰度图像
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 修改最后一个全连接层以输出7种情绪类别
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

class EmotionEfficientNet(nn.Module):
    """
    基于EfficientNet的情感识别模型
    
    使用预训练的EfficientNet-B0作为特征提取器
    输入：灰度图像 (1, 48, 48)
    输出：7种情绪的概率分布
    """
    def __init__(self, num_classes=7):
        super(EmotionEfficientNet, self).__init__()
        
        # 加载预训练的EfficientNet-B0模型
        self.efficientnet = models.efficientnet_b0(weights='DEFAULT')
        
        # 修改第一个卷积层以接受灰度图像
        self.efficientnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # 修改最后一个全连接层以输出7种情绪类别
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)

# 模型工厂函数
def get_emotion_model(model_name, num_classes=7):
    """
    获取指定的情感识别模型
    
    Args:
        model_name: 模型名称，支持 'EmotionCNN', 'DeepEmotionCNN', 'EmotionResNet', 'EmotionEfficientNet'
        num_classes: 情感类别数，默认为7
        
    Returns:
        model: PyTorch模型
    """
    if model_name == 'EmotionCNN':
        return EmotionCNN()
    elif model_name == 'DeepEmotionCNN':
        return DeepEmotionCNN()
    elif model_name == 'EmotionResNet':
        return EmotionResNet(num_classes=num_classes)
    elif model_name == 'EmotionEfficientNet':
        return EmotionEfficientNet(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建测试输入
    x = torch.randn(4, 1, 48, 48).to(device)
    
    # 测试EmotionCNN
    model = EmotionCNN().to(device)
    print(f"EmotionCNN输出形状: {model(x).shape}")
    
    # 测试DeepEmotionCNN
    model = DeepEmotionCNN().to(device)
    print(f"DeepEmotionCNN输出形状: {model(x).shape}")
    
    # 测试EmotionResNet
    model = EmotionResNet().to(device)
    print(f"EmotionResNet输出形状: {model(x).shape}")
    
    # 测试EmotionEfficientNet
    model = EmotionEfficientNet().to(device)
    print(f"EmotionEfficientNet输出形状: {model(x).shape}")
    
    # 计算模型参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"EmotionCNN参数量: {count_parameters(EmotionCNN().to(device)):,}")
    print(f"DeepEmotionCNN参数量: {count_parameters(DeepEmotionCNN().to(device)):,}")
    print(f"EmotionResNet参数量: {count_parameters(EmotionResNet().to(device)):,}")
    print(f"EmotionEfficientNet参数量: {count_parameters(EmotionEfficientNet().to(device)):,}") 