"""
日志分析工具 - 用于对比不同模型的训练性能
"""
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import numpy as np

def parse_filename(filename):
    """从文件名解析模型参数"""
    basename = os.path.basename(filename)
    
    # CNN简化模式: cnn_dp0.1_wd0.011.csv
    simple_cnn_pattern = r'cnn_dp(\d+\.\d+)_wd(\d+\.\d+)'
    simple_cnn_match = re.match(simple_cnn_pattern, basename)
    if (simple_cnn_match):
        return {
            'model': 'cnn',
            'dropout': float(simple_cnn_match.group(1)),
            'weight_decay': float(simple_cnn_match.group(2)),
            # 为缺失参数设置默认值
            'batch_size': 64,  # 默认值
            'hidden_dim': 128,  # 默认值
            'embedding_dim': 100,  # 默认值
            'channels': 64,  # 默认值
            'kernel_size': 3   # 默认值
        }
    
    # LSTM简化模式: lstm_dp0.3_wd0.001.csv 或 lstmattention_dp0.3_wd0.001.csv
    simple_lstm_pattern = r'(lstm|lstmattention|bilstm|bilstmattention)_dp(\d+\.\d+)_wd(\d+\.\d+)'
    simple_lstm_match = re.match(simple_lstm_pattern, basename)
    if (simple_lstm_match):
        return {
            'model': simple_lstm_match.group(1),
            'dropout': float(simple_lstm_match.group(2)),
            'weight_decay': float(simple_lstm_match.group(3)),
            # 为缺失参数设置默认值
            'batch_size': 64,  # 默认值
            'hidden_dim': 128,  # 默认值
            'embedding_dim': 100,  # 默认值
        }
    
    # CNN特定参数模式: cnn_ch64_k3_bs64_dp0.50_hd128_ed100_wd1e-04.csv
    cnn_pattern = r'cnn_ch(\d+)_k(\d+)_bs(\d+)_dp(\d+\.\d+)_hd(\d+)_ed(\d+)_wd(\d+e-\d+)'
    cnn_match = re.match(cnn_pattern, basename)
    if (cnn_match):
        return {
            'model': 'cnn',
            'channels': int(cnn_match.group(1)),
            'kernel_size': int(cnn_match.group(2)),
            'batch_size': int(cnn_match.group(3)),
            'dropout': float(cnn_match.group(4)),
            'hidden_dim': int(cnn_match.group(5)),
            'embedding_dim': int(cnn_match.group(6)),
            'weight_decay': float(cnn_match.group(7))
        }
    
    # 基本模式: model_bs64_dp0.50_hd128_ed100_wd1e-04.csv
    base_pattern = r'([a-z_]+)_bs(\d+)_dp(\d+\.\d+)_hd(\d+)_ed(\d+)_wd(\d+e-\d+)'
    base_match = re.match(base_pattern, basename)
    if (base_match):
        return {
            'model': base_match.group(1),
            'batch_size': int(base_match.group(2)),
            'dropout': float(base_match.group(3)),
            'hidden_dim': int(base_match.group(4)),
            'embedding_dim': int(base_match.group(5)),
            'weight_decay': float(base_match.group(6))
        }
    
    # 尝试匹配任何包含模型名称的文件
    fallback_pattern = r'(lstm|lstmattention|bilstm|bilstmattention|cnn|rnn|gru).*'
    fallback_match = re.match(fallback_pattern, basename)
    if (fallback_match):
        print(f"使用后备模式解析文件: {basename}, 模型类型: {fallback_match.group(1)}")
        # 尝试从文件名中提取参数
        dp_match = re.search(r'dp(\d+\.\d+)', basename)
        wd_match = re.search(r'wd(\d+\.\d+)', basename)
        
        return {
            'model': fallback_match.group(1),
            'dropout': float(dp_match.group(1)) if dp_match else 0.5,
            'weight_decay': float(wd_match.group(1)) if wd_match else 0.001,
            'batch_size': 64,  # 默认值
            'hidden_dim': 128,  # 默认值
            'embedding_dim': 100,  # 默认值
        }
    
    print(f"无法解析的文件名格式: {basename}")
    return None

def read_log_files(log_dir):
    """读取所有日志文件并返回数据框列表"""
    log_files = []
    for model_dir in os.listdir(log_dir):
        model_path = os.path.join(log_dir, model_dir)
        if (os.path.isdir(model_path)):
            log_files.extend(glob.glob(os.path.join(model_path, "*.csv")))
    
    print(f"找到 {len(log_files)} 个日志文件")
    
    log_data = []
    for file in log_files:
        params = parse_filename(file)
        if (not params):
            print(f"无法解析文件名: {file}")
            continue
            
        try:
            df = pd.read_csv(file)
            if (df.empty):
                print(f"文件为空: {file}")
                continue
                
            # 添加模型参数作为列
            for key, value in params.items():
                df[key] = value
            
            # 添加文件名列，便于追踪
            df['filename'] = os.path.basename(file)
            log_data.append(df)
            print(f"成功加载: {file}, 包含 {len(df)} 条记录")
        except Exception as e:
            print(f"读取文件 {file} 时发生错误: {e}")
            
    return log_data

def create_comparison_plots(log_data, output_dir):
    """创建比较不同模型性能的图表"""
    if (not log_data):
        print("未找到日志数据")
        return
        
    # 合并所有日志数据
    all_logs = pd.concat(log_data, ignore_index=True)
    
    # 获取模型类型列表
    models = all_logs['model'].unique()
    print(f"发现的模型类型: {models}")
    
    # 创建结果文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 对比各指标
    metrics = ['train_acc', 'val_acc', 'train_loss', 'val_loss', 'f1', 'recall']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # 1. 创建模型性能摘要表格
    model_summary = []
    
    for model in models:
        model_data = all_logs[all_logs['model'] == model]
        model_configs = model_data['filename'].unique()
        
        for config in model_configs:
            config_data = model_data[model_data['filename'] == config]
            max_val_acc_idx = config_data['val_acc'].idxmax()
            best_epoch = config_data.loc[max_val_acc_idx]
            
            summary_dict = {
                'model': model,
                'dropout': best_epoch['dropout'],
                'weight_decay': best_epoch['weight_decay'],
                'best_epoch': best_epoch['epoch'],
                'val_acc': best_epoch['val_acc'],
                'train_acc': best_epoch['train_acc'],
                'f1': best_epoch['f1'],
                'recall': best_epoch['recall'],
                'val_loss': best_epoch['val_loss'],
                'filename': config
            }
            
            # 添加可能存在的额外参数
            for param in ['batch_size', 'hidden_dim', 'embedding_dim', 'channels', 'kernel_size']:
                if (param in best_epoch):
                    summary_dict[param] = best_epoch[param]
            
            model_summary.append(summary_dict)
    
    summary_df = pd.DataFrame(model_summary)
    summary_df.to_csv(os.path.join(output_dir, "model_performance_summary.csv"), index=False)
    print(f"模型性能摘要已保存至 {os.path.join(output_dir, 'model_performance_summary.csv')}")
    
    # 2. 为每个指标绘制对比图 - 使用改进的方法减少拥挤
    # 2.1 仅使用每个模型的最佳配置进行整体对比图
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # 获取每个模型的最佳配置
        best_configs = summary_df.loc[summary_df.groupby('model')['val_acc'].idxmax()]
        
        for i, (idx, row) in enumerate(best_configs.iterrows()):
            model = row['model']
            filename = row['filename']
            
            # 获取该配置的完整数据
            config_data = all_logs[(all_logs['model'] == model) & (all_logs['filename'] == filename)]
            config_data = config_data.sort_values('epoch')
            
            # 构建标签，根据可用参数
            if (model == 'cnn'):
                # 简化版CNN标签，只显示dropout和weight_decay
                label = f"dp={row['dropout']:.2f}, wd={row['weight_decay']:.6f}"
                
                # 可选：如果需要分类显示其他参数组合，可以添加前缀
                if ('channels' in row and 'kernel_size' in row):
                    ch = row['channels']
                    k = row['kernel_size']
                    # 只有当这些参数不是所有配置都相同时才添加到标签
                    if (len(best_configs[best_configs['model']=='cnn']['channels'].unique()) > 1):
                        label = f"ch={ch}, k={k}, " + label
            elif (model in ['lstm', 'lstmattention', 'bilstm', 'bilstmattention']):
                # LSTM及其变体的标签
                label = f"{model} (dp={row['dropout']:.2f}, wd={row['weight_decay']:.6f})"
                if ('batch_size' in row and 'hidden_dim' in row):
                    label += f", bs={row.get('batch_size', 'N/A')}, hd={row.get('hidden_dim', 'N/A')}"
            else:
                label = f"{model} (bs={row.get('batch_size', 'N/A')}, dp={row['dropout']:.2f}, hd={row.get('hidden_dim', 'N/A')})"
            
            plt.plot(config_data['epoch'], config_data[metric], 
                     label=label, marker='o', color=colors[i % len(colors)], linewidth=2)
        
        # 针对CNN模型的特别处理 - 添加模型类型到标题中
        title_prefix = ''
        if (any(model == 'cnn' for model in best_configs['model'].values)):
            title_prefix = 'CNN '
            
        plt.title(f'{title_prefix}Best Configuration {metric} Comparison', fontsize=15)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 改进图例显示
        if (any(model == 'cnn' for model in best_configs['model'].values)):
            plt.legend(loc='best', fontsize=10, title="CNN Parameters")
        else:
            plt.legend(loc='best', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_best_config_comparison.png"), dpi=300)
        plt.close()
        print(f"已生成最佳配置 {metric} 对比图")
    
    # 2.2 按模型类型分组创建单独图表
    for model in models:
        model_data = all_logs[all_logs['model'] == model]
        model_configs = model_data['filename'].unique()
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            for j, config in enumerate(model_configs):
                config_data = model_data[model_data['filename'] == config]
                config_data = config_data.sort_values('epoch')
                
                # 根据模型类型生成适当的标签
                if (model == 'cnn'):
                    dp = config_data['dropout'].iloc[0]
                    wd = config_data['weight_decay'].iloc[0]
                    label = f"dp={dp:.2f}, wd={wd:.6f}"
                    
                    # 添加其他参数如果存在
                    if ('channels' in config_data.columns and 'kernel_size' in config_data.columns):
                        ch = config_data['channels'].iloc[0]
                        k = config_data['kernel_size'].iloc[0]
                        label = f"ch={ch}, k={k}, " + label
                    
                    if ('batch_size' in config_data.columns and 'hidden_dim' in config_data.columns):
                        bs = config_data['batch_size'].iloc[0]
                        hd = config_data['hidden_dim'].iloc[0]
                        label += f", bs={bs}, hd={hd}"
                elif (model in ['lstm', 'lstmattention', 'bilstm', 'bilstmattention']):
                    # LSTM及其变体的简化标签
                    dp = config_data['dropout'].iloc[0]
                    wd = config_data['weight_decay'].iloc[0]
                    label = f"dp={dp:.2f}, wd={wd:.6f}"
                    
                    # 添加其他参数如果存在
                    if ('batch_size' in config_data.columns and 'hidden_dim' in config_data.columns):
                        bs = config_data['batch_size'].iloc[0]
                        hd = config_data['hidden_dim'].iloc[0]
                        label += f", bs={bs}, hd={hd}"
                else:
                    # 提取标准参数用于标签
                    bs = config_data['batch_size'].iloc[0] if 'batch_size' in config_data.columns else 'N/A'
                    dp = config_data['dropout'].iloc[0]
                    hd = config_data['hidden_dim'].iloc[0] if 'hidden_dim' in config_data.columns else 'N/A'
                    ed = config_data['embedding_dim'].iloc[0] if 'embedding_dim' in config_data.columns else 'N/A'
                    label = f"bs={bs}, dp={dp:.2f}, hd={hd}, ed={ed}"
                
                plt.plot(config_data['epoch'], config_data[metric], 
                         label=label, marker='o', color=colors[j % len(colors)])
            
            plt.title(f'{model} - {metric}', fontsize=15)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(loc='best', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model}_{metric}_comparison.png"), dpi=300)
            plt.close()
            
        print(f"已生成 {model} 模型的各指标对比图")
    
    # 2.3 创建Accuracy和Loss的组合图
    # 为每个模型创建一个包含训练/验证准确率和损失的组合图
    for model in models:
        # 获取该模型的最佳配置
        model_best_config = summary_df[summary_df['model'] == model].sort_values('val_acc', ascending=False).iloc[0]
        filename = model_best_config['filename']
        
        # 获取该配置的完整数据
        config_data = all_logs[(all_logs['model'] == model) & (all_logs['filename'] == filename)]
        config_data = config_data.sort_values('epoch')
        
        # 创建2x2子图布局
        fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # 准确率图
        axs[0].plot(config_data['epoch'], config_data['train_acc'], 'b-o', label='Train Accuracy')
        axs[0].plot(config_data['epoch'], config_data['val_acc'], 'r-^', label='Validation Accuracy')
        axs[0].set_ylabel('Accuracy (%)')
        axs[0].set_title(f'{model} - Accuracy (bs={model_best_config["batch_size"]}, dp={model_best_config["dropout"]:.1f})')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].legend()
        
        # 损失图
        axs[1].plot(config_data['epoch'], config_data['train_loss'], 'b-o', label='Train Loss')
        axs[1].plot(config_data['epoch'], config_data['val_loss'], 'r-^', label='Validation Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_title(f'{model} - Loss')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend()
        
        # 确保X轴是整数
        for ax in axs:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model}_acc_loss.png"), dpi=300)
        plt.close()
        
        print(f"已生成 {model} 模型的准确率/损失组合图")
        
    # 3. 生成模型最佳性能对比图表
    best_model_data = summary_df.sort_values('val_acc', ascending=False)
    
    plt.figure(figsize=(14, 8))
    metrics_to_plot = ['val_acc', 'train_acc', 'f1', 'recall']
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        # 修复 seaborn 使用方式：添加 hue 参数并设置 legend=False
        sns.barplot(x='model', y=metric, hue='model', data=best_model_data, palette='viridis', legend=False)
        plt.title(f'Best {metric} by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "best_model_comparison.png"), dpi=300)
    plt.close()
    print(f"已生成最佳模型对比图")
    
    # 4. 模型参数影响热力图（以val_acc为指标）
    for model in models:
        model_specific = summary_df[summary_df['model'] == model]
        if (len(model_specific) > 3):  # 确保有足够的数据点
            plt.figure(figsize=(10, 8))
            pivot_data = model_specific.pivot_table(
                values='val_acc', 
                index='dropout',
                columns='hidden_dim',
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.2f')
            plt.title(f'{model} Validation Accuracy by Dropout and Hidden Dimension')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model}_param_heatmap.png"), dpi=300)
            plt.close()
            print(f"已生成 {model} 参数热力图")
            
            # 专门为CNN模型创建dropout vs weight_decay热力图
            if (model == 'cnn'):
                plt.figure(figsize=(10, 8))
                # 创建dropout vs weight_decay的数据透视表
                try:
                    dropout_wd_pivot = model_specific.pivot_table(
                        values='val_acc',
                        index='dropout',
                        columns='weight_decay',
                        aggfunc='mean'
                    )
                    # 确保数据足够才绘图
                    if (not dropout_wd_pivot.empty and dropout_wd_pivot.shape[0] > 1 and dropout_wd_pivot.shape[1] > 1):
                        sns.heatmap(dropout_wd_pivot, annot=True, cmap='YlGnBu', fmt='.2f')
                        plt.title('CNN Validation Accuracy: Dropout vs Weight Decay')
                        plt.xlabel('Weight Decay')
                        plt.ylabel('Dropout')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, "cnn_dropout_wd_heatmap.png"), dpi=300)
                        print("已生成CNN dropout vs weight_decay热力图")
                    else:
                        print("CNN数据不足以生成dropout vs weight_decay热力图")
                except Exception as e:
                    print(f"生成CNN dropout vs weight_decay热力图时出错: {e}")
                finally:
                    plt.close()

def generate_html_report(output_dir):
    """生成HTML格式的报告，展示所有生成的图表和分析结果"""
    summary_file = os.path.join(output_dir, "model_performance_summary.csv")
    if (not os.path.exists(summary_file)):
        print("未找到性能摘要文件，无法生成HTML报告")
        return
        
    summary_df = pd.read_csv(summary_file)
    
    # 找出每个模型的最佳配置
    best_configs = summary_df.loc[summary_df.groupby('model')['val_acc'].idxmax()]
    best_overall = summary_df.loc[summary_df['val_acc'].idxmax()]
    
    # 生成HTML报告
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>模型训练性能对比分析</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric-img {{ max-width: 100%; margin-top: 10px; }}
        .container {{ display: flex; flex-wrap: wrap; }}
        .chart {{ margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 15px; }}
        .highlight {{ background-color: #e8f4f8; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>模型训练性能对比分析报告</h1>
    
    <h2>最佳性能模型</h2>
    <p>整体最佳模型: <strong>{best_overall['model']} (验证准确率: {best_overall['val_acc']:.2f}%)</strong></p>
    """
    
    # 根据模型类型显示不同的参数
    if (best_overall['model'] == 'cnn'):
        html_content += f"""<p>参数配置: dropout={best_overall['dropout']:.2f}, weight_decay={best_overall['weight_decay']:.6f}</p>"""
        # 显示其他参数如果存在
        if ('channels' in best_overall and 'kernel_size' in best_overall):
            html_content += f"""<p>CNN参数: channels={best_overall.get('channels', 'N/A')}, kernel_size={best_overall.get('kernel_size', 'N/A')}</p>"""
        if ('batch_size' in best_overall and 'hidden_dim' in best_overall):
            html_content += f"""<p>其他参数: batch_size={best_overall.get('batch_size', 'N/A')}, hidden_dim={best_overall.get('hidden_dim', 'N/A')}, 
                embedding_dim={best_overall.get('embedding_dim', 'N/A')}</p>"""
    elif (best_overall['model'] in ['lstm', 'lstmattention', 'bilstm', 'bilstmattention']):
        html_content += f"""<p>参数配置: dropout={best_overall['dropout']:.2f}, weight_decay={best_overall['weight_decay']:.6f}</p>"""
        if ('batch_size' in best_overall and 'hidden_dim' in best_overall):
            html_content += f"""<p>其他参数: batch_size={best_overall.get('batch_size', 'N/A')}, hidden_dim={best_overall.get('hidden_dim', 'N/A')}, 
                embedding_dim={best_overall.get('embedding_dim', 'N/A')}</p>"""
    else:
        html_content += f"""<p>参数配置: batch_size={best_overall.get('batch_size', 'N/A')}, dropout={best_overall['dropout']:.2f}, 
        hidden_dim={best_overall.get('hidden_dim', 'N/A')}, embedding_dim={best_overall.get('embedding_dim', 'N/A')}</p>"""
    
    html_content += """
    
    <h2>各模型最佳配置</h2>
    <table>
        <tr>
            <th>模型</th>
            <th>验证准确率</th>
            <th>训练准确率</th>
            <th>F1分数</th>
            <th>召回率</th>
            <th>Batch Size</th>
            <th>Dropout</th>
            <th>Hidden Dim</th>
            <th>Embedding Dim</th>
            <th>最佳Epoch</th>
        </tr>
    """
    
    for _, row in best_configs.iterrows():
        html_content += f"""
        <tr class="{'highlight' if row['model'] == best_overall['model'] else ''}">
            <td>{row['model']}</td>
            <td>{row['val_acc']:.2f}%</td>
            <td>{row['train_acc']:.2f}%</td>
            <td>{row['f1']:.2f}%</td>
            <td>{row['recall']:.2f}%</td>
            <td>{row['batch_size']}</td>
            <td>{row['dropout']}</td>
            <td>{row['hidden_dim']}</td>
            <td>{row['embedding_dim']}</td>
            <td>{row['best_epoch']}</td>
        </tr>
        """
    
    html_content += """
    </table>
    
    <h2>性能指标对比图</h2>
    <div class="container">
    """
    
    metrics = ['val_acc', 'train_acc', 'val_loss', 'train_loss', 'f1', 'recall']
    for metric in metrics:
        chart_path = os.path.join(output_dir, f"{metric}_comparison.png")
        if (os.path.exists(chart_path)):
            relative_path = os.path.basename(chart_path)
            html_content += f"""
            <div class="chart">
                <h3>{metric} 对比</h3>
                <img src="{relative_path}" alt="{metric} comparison" class="metric-img">
            </div>
            """
    
    html_content += """
    </div>
    
    <h2>最佳模型对比</h2>
    <img src="best_model_comparison.png" alt="Best Model Comparison" style="max-width: 100%;">
    
    <h2>参数影响分析</h2>
    <div class="container">
    """
    
    # 添加参数热力图
    for file in os.listdir(output_dir):
        if (file.endswith("_param_heatmap.png")):
            model_name = file.split("_param_heatmap.png")[0]
            html_content += f"""
            <div class="chart">
                <h3>{model_name} 参数影响</h3>
                <img src="{file}" alt="{model_name} Parameter Analysis" class="metric-img">
            </div>
            """
    
    html_content += """
    </div>
    
    <h2>结论与建议</h2>
    <p>根据性能对比分析，我们得出以下结论：</p>
    <ul>
    """
    
    # 添加简单结论
    models_sorted = best_configs.sort_values('val_acc', ascending=False)
    html_content += f"""
        <li>最佳性能模型是 <strong>{best_overall['model']}</strong>，验证准确率达到 {best_overall['val_acc']:.2f}%</li>
        <li>模型性能排名（按验证准确率）：{', '.join([f"{row['model']} ({row['val_acc']:.2f}%)" for _, row in models_sorted.iterrows()])}</li>
    """
    
    # 根据最佳模型类型添加不同的结论
    if (best_overall['model'] == 'cnn'):
        html_content += f"""
        <li>对于 {best_overall['model']} 模型，最佳超参数配置为：dropout={best_overall['dropout']:.2f}, 
            weight_decay={best_overall['weight_decay']:.6f}</li>
        """
    elif (best_overall['model'] in ['lstm', 'lstmattention', 'bilstm', 'bilstmattention']):
        html_content += f"""
        <li>对于 {best_overall['model']} 模型，最佳超参数配置为：dropout={best_overall['dropout']:.2f}, 
            weight_decay={best_overall['weight_decay']:.6f}</li>
        """
    else:
        html_content += f"""
        <li>对于 {best_overall['model']} 模型，最佳超参数配置为：batch_size={best_overall.get('batch_size', 'N/A')}, 
            dropout={best_overall['dropout']:.2f}, hidden_dim={best_overall.get('hidden_dim', 'N/A')}, 
            embedding_dim={best_overall.get('embedding_dim', 'N/A')}</li>
        """
    
    html_content += """
    </ul>
    
    <footer>
        <p>此报告由模型对比分析工具自动生成</p>
        <p>生成时间: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </footer>
</body>
</html>
    """
    
    with open(os.path.join(output_dir, "model_comparison_report.html"), "w") as f:
        f.write(html_content)
    
    print(f"HTML报告已生成至 {os.path.join(output_dir, 'model_comparison_report.html')}")

if __name__ == "__main__":
    # 设置路径
    log_dir = "log"
    output_dir = "model_comparison_results"
    
    print("开始分析日志文件...")
    log_data = read_log_files(log_dir)
    
    if (log_data):
        print(f"成功读取 {len(log_data)} 个日志文件的数据，开始生成对比图表...")
        create_comparison_plots(log_data, output_dir)
        generate_html_report(output_dir)
        print(f"分析完成！所有结果已保存至 {output_dir} 文件夹")
    else:
        print("未找到有效的日志数据，请检查日志文件夹路径是否正确")
