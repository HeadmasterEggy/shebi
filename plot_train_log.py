import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.ndimage import gaussian_filter1d


def plot_training_log(model_name=None, save_path=None, compare=False):
    """
    绘制训练日志曲线，支持单模型或模型比较
    
    参数：
        model_name: 模型名称 ('lstm' 或 'cnn')，如果为None且compare=False，则使用默认日志名
        save_path: 保存图片的路径，如果为None则自动生成
        compare: 是否比较两个模型 (lstm vs cnn)
    """
    if compare:
        # 比较模式：加载两个模型的数据
        models = ['lstm', 'cnn']
        dfs = {}
        
        for model in models:
            log_path = f"train_log_{model}.csv"
            try:
                df = pd.read_csv(log_path)
                # 自动识别并按 epoch 聚合（兼容 step 级日志）
                if df["epoch"].duplicated().any():
                    print(f"📌 检测到 {model} 每step日志，自动按epoch聚合...")
                    df = df.groupby("epoch", as_index=False).mean()
                dfs[model] = df
            except Exception as e:
                print(f"⚠️ 无法加载 {model} 模型日志: {e}")
                return
        
        if not save_path:
            save_path = "training_metrics_comparison.png"
        
        plot_comparison(dfs, save_path)
    else:
        # 单模型模式
        model_suffix = f"_{model_name}" if model_name else ""
        log_path = f"train_log{model_suffix}.csv"
        
        if not save_path:
            model_info = f"_{model_name}" if model_name else ""
            save_path = f"training_metrics{model_info}.png"
        
        try:
            df = pd.read_csv(log_path)
        except Exception as e:
            print(f"⚠️ 无法加载训练日志 {log_path}: {e}")
            return
            
        # 自动识别并按 epoch 聚合（兼容 step 级日志）
        if df["epoch"].duplicated().any():
            print("📌 检测到每step日志，自动按epoch聚合...")
            df = df.groupby("epoch", as_index=False).mean()

        plot_single_model(df, save_path, model_name)


def plot_single_model(df, save_path, model_name=None):
    """为单个模型绘制训练指标图"""
    epochs = df["epoch"]
    smooth_sigma = 1.5

    def smooth_or_raw(col):
        return gaussian_filter1d(df[col], sigma=smooth_sigma) if col in df else None

    loss_s = smooth_or_raw("train_loss")
    acc_s = smooth_or_raw("train_acc")
    f1_s = smooth_or_raw("f1")
    recall_s = smooth_or_raw("recall")

    val_loss_s = smooth_or_raw("val_loss")
    val_acc_s = smooth_or_raw("val_acc")
    val_f1_s = smooth_or_raw("val_f1")
    val_recall_s = smooth_or_raw("val_recall")

    plt.figure(figsize=(16, 10))
    plt.style.use("seaborn-v0_8-muted")
    
    model_title = f" ({model_name.upper()})" if model_name else ""

    # --- Loss ---
    plt.subplot(2, 2, 1)
    if loss_s is not None:
        plt.plot(epochs, loss_s, label="Train Loss", color="#E74C3C", linewidth=2)
        plt.scatter(epochs, df["train_loss"], color="#E74C3C", s=20, alpha=0.3)
    if val_loss_s is not None:
        plt.plot(epochs, val_loss_s, label="Val Loss", color="#8E44AD", linestyle="--", linewidth=2)
        plt.scatter(epochs, df["val_loss"], color="#8E44AD", s=20, alpha=0.3)
    plt.title(f"Loss Curve{model_title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # --- Accuracy ---
    plt.subplot(2, 2, 2)
    if acc_s is not None:
        plt.plot(epochs, acc_s, label="Train Acc", color="#3498DB", linewidth=2)
        plt.scatter(epochs, df["train_acc"], color="#3498DB", s=20, alpha=0.3)
        
        # Highlight max train accuracy
        max_train_idx = df["train_acc"].idxmax()
        max_train_epoch = df.loc[max_train_idx, "epoch"]
        max_train_acc = df.loc[max_train_idx, "train_acc"]
        plt.scatter(max_train_epoch, max_train_acc, color="#3498DB", s=100, edgecolor='black', zorder=5)
        plt.annotate(f"Max: {max_train_acc:.2f}%", 
                    (max_train_epoch, max_train_acc),
                    textcoords="offset points", 
                    xytext=(-15, 10), 
                    ha='center',
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
    if val_acc_s is not None:
        plt.plot(epochs, val_acc_s, label="Val Acc", color="#34495E", linestyle="--", linewidth=2)
        plt.scatter(epochs, df["val_acc"], color="#34495E", s=20, alpha=0.3)
        
        # Highlight max validation accuracy
        max_val_idx = df["val_acc"].idxmax()
        max_val_epoch = df.loc[max_val_idx, "epoch"]
        max_val_acc = df.loc[max_val_idx, "val_acc"]
        plt.scatter(max_val_epoch, max_val_acc, color="#34495E", s=100, edgecolor='black', zorder=5)
        plt.annotate(f"Max: {max_val_acc:.2f}%", 
                    (max_val_epoch, max_val_acc),
                    textcoords="offset points", 
                    xytext=(15, -15), 
                    ha='center',
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
    plt.title(f"Accuracy{model_title}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    # --- F1 ---
    plt.subplot(2, 2, 3)
    if f1_s is not None:
        plt.plot(epochs, f1_s, label="Train F1", color="#2ECC71", linewidth=2)
        plt.scatter(epochs, df["f1"], color="#2ECC71", s=20, alpha=0.3)
    if val_f1_s is not None:
        plt.plot(epochs, val_f1_s, label="Val F1", color="#1ABC9C", linestyle="--", linewidth=2)
        plt.scatter(epochs, df["val_f1"], color="#1ABC9C", s=20, alpha=0.3)
    plt.title(f"F1 Score{model_title}")
    plt.xlabel("Epoch")
    plt.ylabel("F1 (%)")
    plt.grid(True)
    plt.legend()

    # --- Recall ---
    plt.subplot(2, 2, 4)
    if recall_s is not None:
        plt.plot(epochs, recall_s, label="Train Recall", color="#F1C40F", linewidth=2)
        plt.scatter(epochs, df["recall"], color="#F1C40F", s=20, alpha=0.3)
    if val_recall_s is not None:
        plt.plot(epochs, val_recall_s, label="Val Recall", color="#D35400", linestyle="--", linewidth=2)
        plt.scatter(epochs, df["val_recall"], color="#D35400", s=20, alpha=0.3)
    plt.title(f"Recall{model_title}")
    plt.xlabel("Epoch")
    plt.ylabel("Recall (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout(h_pad=2)
    plt.savefig(save_path, dpi=150)
    print(f"✅ 训练图已保存至：{save_path}")
    plt.close()


def plot_comparison(dfs, save_path):
    """比较两个模型的训练指标"""
    plt.figure(figsize=(16, 10))
    plt.style.use("seaborn-v0_8-muted")
    
    # 定义每个模型的颜色
    colors = {
        'lstm': {'train': '#3498DB', 'val': '#2874A6'},  # 蓝色系
        'cnn': {'train': '#E74C3C', 'val': '#922B21'}    # 红色系
    }
    
    # 指标名称和它们的位置
    metrics = {
        'Loss': {'pos': 1, 'col': 'train_loss', 'val_col': 'val_loss', 'ylabel': 'Loss'},
        'Accuracy': {'pos': 2, 'col': 'train_acc', 'val_col': 'val_acc', 'ylabel': 'Accuracy (%)'},
        'F1': {'pos': 3, 'col': 'f1', 'val_col': 'val_f1', 'ylabel': 'F1 (%)'},
        'Recall': {'pos': 4, 'col': 'recall', 'val_col': 'val_recall', 'ylabel': 'Recall (%)'}
    }
    
    # 为每个指标创建子图
    for metric_name, metric_info in metrics.items():
        plt.subplot(2, 2, metric_info['pos'])
        
        # 绘制每个模型的指标
        for model_name, df in dfs.items():
            epochs = df["epoch"]
            train_col = metric_info['col']
            val_col = metric_info['val_col']
            
            # 检查列是否存在
            if train_col in df.columns:
                # 平滑处理
                train_data = gaussian_filter1d(df[train_col], sigma=1.5)
                plt.plot(epochs, train_data, 
                         label=f"{model_name.upper()} Train", 
                         color=colors[model_name]['train'],
                         linewidth=2)
                plt.scatter(epochs, df[train_col], 
                           color=colors[model_name]['train'], s=20, alpha=0.3)
                
                # For accuracy metric, highlight the maximum point
                if metric_name == 'Accuracy':
                    max_idx = df[train_col].idxmax()
                    max_epoch = df.loc[max_idx, "epoch"]
                    max_val = df.loc[max_idx, train_col]
                    plt.scatter(max_epoch, max_val, color=colors[model_name]['train'], s=100, edgecolor='black', zorder=5)
                    plt.annotate(f"{model_name.upper()}: {max_val:.2f}%", 
                                (max_epoch, max_val),
                                textcoords="offset points", 
                                xytext=(0, 10), 
                                ha='center',
                                fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # 检查验证列是否存在
            if val_col in df.columns:
                val_data = gaussian_filter1d(df[val_col], sigma=1.5)
                plt.plot(epochs, val_data, 
                         label=f"{model_name.upper()} Val", 
                         color=colors[model_name]['val'],
                         linewidth=2, linestyle='--')
                plt.scatter(epochs, df[val_col], 
                           color=colors[model_name]['val'], s=20, alpha=0.3)
                
                # For accuracy metric, highlight the maximum point
                if metric_name == 'Accuracy':
                    max_val_idx = df[val_col].idxmax()
                    max_val_epoch = df.loc[max_val_idx, "epoch"]
                    max_val_acc = df.loc[max_val_idx, val_col]
                    plt.scatter(max_val_epoch, max_val_acc, color=colors[model_name]['val'], s=100, edgecolor='black', zorder=5)
                    plt.annotate(f"{model_name.upper()} Val: {max_val_acc:.2f}%", 
                                (max_val_epoch, max_val_acc),
                                textcoords="offset points", 
                                xytext=(0, -15), 
                                ha='center',
                                fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.title(f"{metric_name} Comparison")
        plt.xlabel("Epoch")
        plt.ylabel(metric_info['ylabel'])
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout(h_pad=2)
    plt.savefig(save_path, dpi=150)
    print(f"✅ 模型比较图已保存至：{save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制训练日志')
    parser.add_argument('--model', type=str, choices=['lstm', 'cnn'], help='指定模型类型')
    parser.add_argument('--output', type=str, help='输出图像路径')
    parser.add_argument('--compare', action='store_true', help='比较LSTM和CNN模型')
    args = parser.parse_args()
    
    plot_training_log(model_name=args.model, save_path=args.output, compare=args.compare)
