import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def plot_training_log(log_path="train_log.csv", save_path="training_metrics_with_val.png"):
    df = pd.read_csv(log_path)

    # 自动识别并按 epoch 聚合（兼容 step 级日志）
    if df["epoch"].duplicated().any():
        print("📌 检测到每step日志，自动按epoch聚合...")
        df = df.groupby("epoch", as_index=False).mean()

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

    # --- Loss ---
    plt.subplot(2, 2, 1)
    if loss_s is not None:
        plt.plot(epochs, loss_s, label="Train Loss", color="#E74C3C", linewidth=2)
        plt.scatter(epochs, df["train_loss"], color="#E74C3C", s=20, alpha=0.3)
    if val_loss_s is not None:
        plt.plot(epochs, val_loss_s, label="Val Loss", color="#8E44AD", linestyle="--", linewidth=2)
        plt.scatter(epochs, df["val_loss"], color="#8E44AD", s=20, alpha=0.3)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # --- Accuracy ---
    plt.subplot(2, 2, 2)
    if acc_s is not None:
        plt.plot(epochs, acc_s, label="Train Acc", color="#3498DB", linewidth=2)
        plt.scatter(epochs, df["train_acc"], color="#3498DB", s=20, alpha=0.3)
    if val_acc_s is not None:
        plt.plot(epochs, val_acc_s, label="Val Acc", color="#34495E", linestyle="--", linewidth=2)
        plt.scatter(epochs, df["val_acc"], color="#34495E", s=20, alpha=0.3)
    plt.title("Accuracy")
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
    plt.title("F1 Score")
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
    plt.title("Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout(h_pad=2)
    plt.savefig(save_path, dpi=150)
    print(f"✅ 训练 + 验证图已保存至：{save_path}")
    plt.show()

if __name__ == "__main__":
    plot_training_log()
