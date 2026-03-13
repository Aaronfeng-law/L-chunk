#!/usr/bin/env python3
"""
BERT 層級符號分類器訓練器 & 傳統模型比較器
基於  "單一職責" 原則：專注訓練和比較

專注於：
1. 加載標註數據
2. 訓練 BERT 模型
3. 比較傳統機器學習模型 (Logistic Regression, SVM, Random Forest)
4. 保存模型和評估結果
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# BERT相關
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch.nn.functional as F

class BERTLevelSymbolTrainer:
    """BERT 層級符號分類器訓練器"""
    
    def __init__(self, output_dir: str = "models/training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型相關
        self.bert_model = None
        self.bert_tokenizer = None
        self.trainer = None
        
    def train_classifier(self, training_data_path: str) -> Dict:
        """訓練 BERT 分類器"""
        print("🤖 訓練 BERT 層級符號分類器...")
        
        # 載入訓練數據
        df = pd.read_csv(training_data_path)
        print(f"✅ 載入 {len(df)} 條訓練數據")
        
        # 數據統計
        positive_count = (df['sentiment'] == 'Positive').sum()
        negative_count = (df['sentiment'] == 'Negative').sum()
        print(f"📊 正樣本: {positive_count} ({positive_count/len(df)*100:.1f}%)")
        print(f"📊 負樣本: {negative_count} ({negative_count/len(df)*100:.1f}%)")
        
        # 準備數據
        X = df['line_text'].values
        y = (df['sentiment'] == 'Positive').astype(int).values
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 訓練集: {len(X_train)} 樣本")
        print(f"📊 測試集: {len(X_test)} 樣本")
        
        # 初始化 BERT 模型
        model_name = "bert-base-chinese"
        print(f"🔧 初始化 {model_name} 模型...")
        
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
        
        # 準備數據集
        def create_dataset(texts, labels):
            dataset = Dataset.from_dict({
                'text': texts,
                'labels': labels
            })
            return dataset.map(self._tokenize_function, batched=True)
        
        train_dataset = create_dataset(X_train, y_train)
        test_dataset = create_dataset(X_test, y_test)
        
        # 訓練參數 - 基於 text_classifier.py 的優化配置
        print("⚙️ 配置訓練參數...")
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=2,  # 避免過擬合
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=self.output_dir / 'logs',
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            logging_steps=50,
            report_to=[],  # 禁用 wandb 等外部日誌
        )
        
        # 數據整理器
        data_collator = DataCollatorWithPadding(tokenizer=self.bert_tokenizer)
        
        # Trainer
        self.trainer = Trainer(
            model=self.bert_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.bert_tokenizer,
            data_collator=data_collator,
        )
        
        # 訓練
        print("🔥 開始訓練...")
        self.trainer.train()
        
        # 評估
        print("📊 評估模型...")
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_pred_proba = F.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
        
        # 評估指標
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"✅ BERT 訓練完成!")
        print(f"   準確率: {accuracy:.4f}")
        print(f"   精確率: {precision:.4f}")
        print(f"   召回率: {recall:.4f}")
        print(f"   F1分數: {f1:.4f}")
        
        # 保存模型
        best_model_path = self.output_dir / 'best_model'
        print(f"💾 保存模型到: {best_model_path}")
        
        self.trainer.save_model(best_model_path)
        self.bert_tokenizer.save_pretrained(best_model_path)
        
        # 保存訓練信息
        training_info = {
            'model_name': model_name,
            'training_data': training_data_path,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'trained_at': datetime.now().isoformat(),
            'model_path': str(best_model_path)
        }
        
        info_file = self.output_dir / 'training_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        # 保存詳細分類報告
        report = classification_report(y_test, y_pred, target_names=['普通文本', '層級符號'])
        report_file = self.output_dir / 'classification_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📋 訓練信息已保存: {info_file}")
        print(f"📋 分類報告已保存: {report_file}")
        
        return training_info
    
    def _tokenize_function(self, examples):
        """BERT tokenization"""
        return self.bert_tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=512
        )

def main():
    """主函數 - 只負責 BERT 訓練"""
    print("� 啟動 BERT 層級符號分類器訓練")
    print("基於  '單一職責' 原則：專注訓練")
    print("="*60)
    
    # 檢查訓練數據
    training_data = "data/training/project-1-at-2025-10-10-15-05-fea45fba.csv"
    if not Path(training_data).exists():
        print(f"❌ 找不到訓練數據: {training_data}")
        print("請確保標註數據文件存在")
        return
    
    # 初始化訓練器
    trainer = BERTLevelSymbolTrainer()
    
    # 執行 BERT 訓練
    training_info = trainer.train_classifier(training_data)
    
    print(f"\n🎉 BERT 訓練完成!")
    print(f"🏆 最佳性能: 準確率 {training_info['accuracy']:.4f}, 召回率 {training_info['recall']:.4f}")
    print(f"💾 模型保存在: {training_info['model_path']}")
    
    # GPU 設備信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 使用 GPU: {gpu_name}")
    else:
        print("💻 使用 CPU 訓練")
    
    def _tokenize_function(self, examples):
        """BERT tokenization"""
        return self.bert_tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=512
        )

def main():
    """主函數"""
    print("🚀 啟動 BERT 層級符號分類器訓練 & 傳統模型比較")
    print("="*60)
    
    # 檢查訓練數據
    training_data = "data/training/project-1-at-2025-10-10-15-05-fea45fba.csv"
    if not Path(training_data).exists():
        print(f"❌ 找不到訓練數據: {training_data}")
        print("請確保標註數據文件存在")
        return
    
    # 初始化訓練器
    trainer = BERTLevelSymbolTrainer()
    
    # 執行 BERT 訓練
    training_info = trainer.train_classifier(training_data)
    
    print(f"\n🎉 BERT 訓練完成!")
    print(f"🏆 最佳性能: 準確率 {training_info['accuracy']:.4f}, 召回率 {training_info['recall']:.4f}")
    print(f"💾 模型保存在: {training_info['model_path']}")
    
    # 比較傳統機器學習模型
    print("\n" + "="*60)
    print("🔬 開始比較傳統機器學習模型")
    comparison_results = trainer.compare_traditional_models(training_data)
    
    # 顯示比較結果
    print("\n📊 模型性能比較:")
    print("-" * 60)
    print(f"{'模型':<20} {'準確率':<8} {'精確率':<8} {'召回率':<8} {'F1分數':<8}")
    print("-" * 60)
    
    for model_name, metrics in comparison_results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f} {metrics['f1_score']:<8.4f}")
    
    # 與 BERT 比較
    bert_metrics = {
        'accuracy': training_info['accuracy'],
        'precision': training_info['precision'],
        'recall': training_info['recall'],
        'f1_score': training_info['f1_score']
    }
    
    print("-" * 60)
    print(f"{'BERT':<20} {bert_metrics['accuracy']:<8.4f} {bert_metrics['precision']:<8.4f} {bert_metrics['recall']:<8.4f} {bert_metrics['f1_score']:<8.4f}")
    
    # GPU 設備信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 使用 GPU: {gpu_name}")
    else:
        print("💻 使用 CPU 訓練")

if __name__ == "__main__":
    main()