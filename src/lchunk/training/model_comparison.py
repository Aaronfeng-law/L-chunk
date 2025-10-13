#!/usr/bin/env python3
"""
模型比較評估工具
基於  "好品味" 原則：簡單而有效的比較

比較模型：
1. 邏輯回歸 (Logistic Regression)
2. 支持向量機 (SVM)  
3. 隨機森林 (Random Forest)
4. BERT 分類器

目標：找到最佳的層級符號檢測模型
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 傳統機器學習
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_auc_score
)

# BERT相關
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ModelComparisonEvaluator:
    """模型比較評估器"""
    
    def __init__(self, output_dir: str = "output/model_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型存儲
        self.models = {}
        self.vectorizer = None
        self.results = {}
        
        # BERT 相關
        self.bert_model = None
        self.bert_tokenizer = None
        
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """載入訓練數據"""
        print("📊 載入數據...")
        df = pd.read_csv(data_path)
        
        X = df['line_text'].values
        y = (df['sentiment'] == 'Positive').astype(int).values
        
        print(f"✅ 載入 {len(df)} 條數據")
        print(f"   正樣本: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
        print(f"   負樣本: {len(y)-np.sum(y)} ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")
        
        return X, y
    
    def prepare_traditional_features(self, X_train: np.ndarray, X_test: np.ndarray):
        """為傳統機器學習模型準備特徵"""
        print("🔧 準備 TF-IDF 特徵...")
        
        # TF-IDF 向量化
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"✅ TF-IDF 特徵維度: {X_train_tfidf.shape[1]}")
        
        return X_train_tfidf, X_test_tfidf
    
    def train_traditional_models(self, X_train_tfidf, y_train, X_test_tfidf, y_test):
        """訓練傳統機器學習模型"""
        print("\n🤖 訓練傳統機器學習模型...")
        
        # 模型配置
        model_configs = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'SVM': SVC(
                random_state=42,
                probability=True,
                class_weight='balanced',
                kernel='rbf'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=10
            )
        }
        
        # 訓練和評估每個模型
        for name, model in model_configs.items():
            print(f"\n🔥 訓練 {name}...")
            
            # 訓練
            model.fit(X_train_tfidf, y_train)
            
            # 預測
            y_pred = model.predict(X_test_tfidf)
            y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
            
            # 評估
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # 保存模型和結果
            self.models[name] = model
            self.results[name] = metrics
            
            print(f"✅ {name} 訓練完成")
            print(f"   準確率: {metrics['accuracy']:.4f}")
            print(f"   召回率: {metrics['recall']:.4f}")
            print(f"   F1分數: {metrics['f1_score']:.4f}")
            print(f"   AUC: {metrics['auc']:.4f}")
    
    def evaluate_bert_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """評估現有的 BERT 模型"""
        print("\n🤖 評估 BERT 模型...")
        
        bert_model_path = "models/training/best_model"
        if not Path(bert_model_path).exists():
            print(f"❌ 找不到 BERT 模型: {bert_model_path}")
            print("請先訓練 BERT 模型")
            return
        
        # 載入 BERT 模型
        print("📥 載入 BERT 模型...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)
        self.bert_model.eval()
        
        # 批量預測
        y_pred_list = []
        y_pred_proba_list = []
        
        batch_size = 16
        for i in range(0, len(X_test), batch_size):
            batch_texts = X_test[i:i+batch_size]
            
            # Tokenization
            inputs = self.bert_tokenizer(
                batch_texts.tolist(),
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 預測
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                
                # 獲取預測結果
                batch_pred = torch.argmax(logits, dim=1).numpy()
                batch_proba = probs[:, 1].numpy()
                
                y_pred_list.extend(batch_pred)
                y_pred_proba_list.extend(batch_proba)
        
        y_pred = np.array(y_pred_list)
        y_pred_proba = np.array(y_pred_proba_list)
        
        # 評估
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        self.results['BERT'] = metrics
        
        print(f"✅ BERT 評估完成")
        print(f"   準確率: {metrics['accuracy']:.4f}")
        print(f"   召回率: {metrics['recall']:.4f}")
        print(f"   F1分數: {metrics['f1_score']:.4f}")
        print(f"   AUC: {metrics['auc']:.4f}")
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba) -> Dict:
        """計算評估指標"""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred)),
            'auc': float(roc_auc_score(y_true, y_pred_proba)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def generate_comparison_report(self):
        """生成比較報告"""
        print("\n📊 生成比較報告...")
        
        # 創建結果 DataFrame
        metrics_df = pd.DataFrame(self.results).T
        metrics_df = metrics_df.drop('confusion_matrix', axis=1)
        
        # 排序（按 F1 分數）
        metrics_df = metrics_df.sort_values('f1_score', ascending=False)
        
        print("\n📋 模型性能比較:")
        print("="*80)
        print(f"{'模型':<20} {'準確率':<10} {'精確率':<10} {'召回率':<10} {'F1分數':<10} {'AUC':<10}")
        print("="*80)
        
        for model_name, row in metrics_df.iterrows():
            print(f"{model_name:<20} {row['accuracy']:<10.4f} {row['precision']:<10.4f} "
                  f"{row['recall']:<10.4f} {row['f1_score']:<10.4f} {row['auc']:<10.4f}")
        
        # 找出最佳模型
        best_model = metrics_df.index[0]
        best_f1 = metrics_df.loc[best_model, 'f1_score']
        
        print("="*80)
        print(f"🏆 最佳模型: {best_model} (F1分數: {best_f1:.4f})")
        
        return metrics_df, best_model
    
    def create_visualizations(self, metrics_df: pd.DataFrame):
        """創建可視化圖表"""
        print("\n📈 創建可視化圖表...")
        
        # 設置圖表樣式
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能比較', fontsize=16, fontweight='bold')
        
        # 1. 條形圖 - 各項指標比較
        ax1 = axes[0, 0]
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        x = np.arange(len(metrics_df))
        width = 0.15
        
        for i, metric in enumerate(metrics_to_plot):
            ax1.bar(x + i*width, metrics_df[metric], width, label=metric.upper())
        
        ax1.set_xlabel('模型')
        ax1.set_ylabel('分數')
        ax1.set_title('各項評估指標比較')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(metrics_df.index, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. F1分數排序
        ax2 = axes[0, 1]
        colors = ['#2E8B57', '#4682B4', '#DAA520', '#DC143C'][:len(metrics_df)]
        bars = ax2.bar(metrics_df.index, metrics_df['f1_score'], color=colors)
        ax2.set_title('F1分數比較')
        ax2.set_ylabel('F1分數')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for bar, value in zip(bars, metrics_df['f1_score']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 3. 精確率 vs 召回率散點圖
        ax3 = axes[1, 0]
        scatter = ax3.scatter(metrics_df['precision'], metrics_df['recall'], 
                             c=metrics_df['f1_score'], cmap='viridis', s=100, alpha=0.8)
        
        for i, model in enumerate(metrics_df.index):
            ax3.annotate(model, (metrics_df.loc[model, 'precision'], 
                               metrics_df.loc[model, 'recall']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('精確率')
        ax3.set_ylabel('召回率')
        ax3.set_title('精確率 vs 召回率')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='F1分數')
        
        # 4. 雷達圖
        ax4 = axes[1, 1]
        ax4.remove()  # 移除原軸
        ax4 = fig.add_subplot(2, 2, 4, projection='polar')
        
        # 選擇最佳的兩個模型進行雷達圖比較
        top_2_models = metrics_df.head(2)
        metrics_radar = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        angles = np.linspace(0, 2*np.pi, len(metrics_radar), endpoint=False).tolist()
        angles += angles[:1]  # 閉合
        
        for i, (model_name, row) in enumerate(top_2_models.iterrows()):
            values = [row[metric] for metric in metrics_radar]
            values += values[:1]  # 閉合
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax4.fill(angles, values, alpha=0.25)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels([metric.upper() for metric in metrics_radar])
        ax4.set_title('前兩名模型雷達圖比較')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存圖表
        chart_path = self.output_dir / 'model_comparison_charts.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"📊 圖表已保存: {chart_path}")
        
        plt.show()
    
    def save_results(self, metrics_df: pd.DataFrame, best_model: str):
        """保存結果"""
        print("\n💾 保存評估結果...")
        
        # 保存詳細結果
        results_file = self.output_dir / f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 保存 CSV 報告
        csv_file = self.output_dir / f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        metrics_df.to_csv(csv_file, encoding='utf-8')
        
        # 保存最佳模型
        if best_model in self.models:
            best_model_file = self.output_dir / f'best_traditional_model_{best_model.replace(" ", "_").lower()}.pkl'
            with open(best_model_file, 'wb') as f:
                pickle.dump({
                    'model': self.models[best_model],
                    'vectorizer': self.vectorizer,
                    'model_name': best_model
                }, f)
            print(f"🏆 最佳傳統模型已保存: {best_model_file}")
        
        # 創建總結報告
        summary_file = self.output_dir / f'comparison_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        self._create_summary_report(summary_file, metrics_df, best_model)
        
        print(f"📋 詳細結果: {results_file}")
        print(f"📊 CSV報告: {csv_file}")
        print(f"📝 總結報告: {summary_file}")
    
    def _create_summary_report(self, file_path: Path, metrics_df: pd.DataFrame, best_model: str):
        """創建總結報告"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# 模型比較評估報告\n\n")
            f.write(f"**評估時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 模型性能總覽\n\n")
            f.write("| 模型 | 準確率 | 精確率 | 召回率 | F1分數 | AUC |\n")
            f.write("|------|--------|--------|--------|--------|----- |\n")
            
            for model_name, row in metrics_df.iterrows():
                f.write(f"| {model_name} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
                       f"{row['recall']:.4f} | {row['f1_score']:.4f} | {row['auc']:.4f} |\n")
            
            f.write(f"\n## 🏆 最佳模型: {best_model}\n\n")
            
            best_metrics = metrics_df.loc[best_model]
            f.write("### 最佳模型性能:\n")
            f.write(f"- **準確率**: {best_metrics['accuracy']:.4f}\n")
            f.write(f"- **精確率**: {best_metrics['precision']:.4f}\n")
            f.write(f"- **召回率**: {best_metrics['recall']:.4f}\n")
            f.write(f"- **F1分數**: {best_metrics['f1_score']:.4f}\n")
            f.write(f"- **AUC**: {best_metrics['auc']:.4f}\n\n")
            
            f.write("## 模型分析\n\n")
            
            traditional_models = [name for name in metrics_df.index if name != 'BERT']
            if 'BERT' in metrics_df.index:
                bert_f1 = metrics_df.loc['BERT', 'f1_score']
                best_traditional_f1 = max([metrics_df.loc[name, 'f1_score'] for name in traditional_models])
            


def main():
    """主函數"""
    print("🚀 啟動模型比較評估")
    print("="*60)
    
    # 檢查數據文件
    data_file = "data/training/project-1-at-2025-10-10-15-05-fea45fba.csv"
    if not Path(data_file).exists():
        print(f"❌ 找不到數據文件: {data_file}")
        return
    
    # 初始化評估器
    evaluator = ModelComparisonEvaluator()
    
    # 載入數據
    X, y = evaluator.load_data(data_file)
    
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 準備傳統機器學習特徵
    X_train_tfidf, X_test_tfidf = evaluator.prepare_traditional_features(X_train, X_test)
    
    # 訓練傳統模型
    evaluator.train_traditional_models(X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # 評估 BERT 模型
    evaluator.evaluate_bert_model(X_test, y_test)
    
    # 生成比較報告
    metrics_df, best_model = evaluator.generate_comparison_report()
    
    # 創建可視化
    evaluator.create_visualizations(metrics_df)
    
    # 保存結果
    evaluator.save_results(metrics_df, best_model)
    
    print(f"\n🎉 評估完成!")
    print(f"🏆 最佳模型: {best_model}")
    print(f"📁 結果保存在: {evaluator.output_dir}")


if __name__ == "__main__":
    main()