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
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置全局字体为宋体
try:
    # 配置matplotlib使用Songti SC字体
    plt.rcParams['font.family'] = ['Songti SC']
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建宋体字体对象，用于后续引用
    songti_font = FontProperties(family='Songti SC')
    
    print("成功设置Songti SC字体")
except:
    print("注意: 无法设置Songti SC字体，将使用系统默认字体")

# 添加一个辅助函数来检测bilstm_attention文件
def is_bilstm_attention_file(filename):
    """检查是否为bilstm_attention模型文件"""
    basename = os.path.basename(filename).lower()
    # 检查多种可能的格式组合
    patterns = [
        'bilstm_attention', 'bilstm-attention', 'bilstm_att', 'bilstm-att',
        'bilstmattention', 'bilstm-attn', 'bilstm_attn', 'bi_lstm_attention'
    ]
    for pattern in patterns:
        if pattern in basename:
            print(f"发现BiLSTM Attention文件: {basename} (匹配模式: {pattern})")
            return True
    return False

def parse_filename(filename):
    """从文件名解析模型参数"""
    basename = os.path.basename(filename)
    print(f"正在解析文件: {basename}")
    
    # 主要匹配模式: {model}_dp{dropout}_wd{weight_decay}.csv
    main_pattern = r'(\w+)_dp(\d+\.\d+)_wd((?:\d+\.\d+)|(?:\d+e-\d+))'
    main_match = re.match(main_pattern, basename)
    
    if main_match:
        model_name = main_match.group(1).lower()
        dropout = float(main_match.group(2))
        weight_decay = float(main_match.group(3))
        
        # 检查是否为attention模型
        if is_bilstm_attention_file(basename):
            model_name = 'bilstm_attention'
        elif 'lstm_attention' in basename.lower() or 'lstm-attention' in basename.lower():
            model_name = 'lstm_attention'
            
        print(f"匹配成功: 模型={model_name}, dropout={dropout}, weight_decay={weight_decay}")
        
        return {
            'model': model_name,
            'dropout': dropout,
            'weight_decay': weight_decay,
            # 为缺失参数设置默认值
            'batch_size': 64,  # 默认值
            'hidden_dim': 128,  # 默认值
            'embedding_dim': 100,  # 默认值
        }
    
    # 备用匹配模式，尝试直接从文件名中提取参数
    model_type = None
    # 检查是否为bilstm_attention文件
    if is_bilstm_attention_file(basename):
        model_type = 'bilstm_attention'
    elif 'lstm_attention' in basename.lower() or 'lstm-attention' in basename.lower():
        model_type = 'lstm_attention'
    elif 'bilstm' in basename.lower():
        model_type = 'bilstm'
    elif 'lstm' in basename.lower():
        model_type = 'lstm'
    elif 'cnn' in basename.lower():
        model_type = 'cnn'
    
    # 尝试从文件名中提取dropout和weight_decay
    dp_match = re.search(r'dp(\d+\.\d+)', basename)
    wd_match = re.search(r'wd((?:\d+\.\d+)|(?:\d+e-\d+))', basename)
    
    if model_type and dp_match and wd_match:
        dropout = float(dp_match.group(1))
        weight_decay = float(wd_match.group(1))
        
        print(f"备用匹配: 模型={model_type}, dropout={dropout}, weight_decay={weight_decay}")
        
        return {
            'model': model_type,
            'dropout': dropout,
            'weight_decay': weight_decay,
            # 为缺失参数设置默认值
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
    
    # 添加日志文件列表的输出，帮助排查问题
    print("日志文件列表:")
    for i, file in enumerate(log_files[:10]):  # 只输出前10个以避免过多
        print(f"  {i+1}. {os.path.basename(file)}")
    if len(log_files) > 10:
        print(f"  ... 以及另外 {len(log_files) - 10} 个文件")
    
    # 按模型类型计数文件
    model_counts = {}
    model_examples = {}
    
    log_data = []
    for file in log_files:
        params = parse_filename(file)
        if (not params):
            print(f"无法解析文件名: {file}")
            continue
        
        model_name = params['model']
        if model_name in model_counts:
            model_counts[model_name] += 1
        else:
            model_counts[model_name] = 1
            model_examples[model_name] = os.path.basename(file)
            
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
            print(f"成功加载: {file}, 包含 {len(df)} 条记录, 模型类型: {model_name}")
        except Exception as e:
            print(f"读取文件 {file} 时发生错误: {e}")
    
    # 打印各模型文件数量统计
    print("\n各模型文件数量统计:")
    for model, count in model_counts.items():
        print(f"  - {model}: {count}个文件, 示例: {model_examples.get(model, 'N/A')}")
    
    return log_data

def create_comparison_plots(log_data, output_dir):
    """创建比较不同模型性能的图表，专注于dropout和weight_decay的比较"""
    if (not log_data):
        print("未找到日志数据")
        return
        
    # 合并所有日志数据
    all_logs = pd.concat(log_data, ignore_index=True)
    
    # 获取模型类型列表并打印模型和数据量
    models = all_logs['model'].unique()
    print(f"\n发现的模型类型: {models}")
    
    # 打印每种模型的数据量
    for model in models:
        model_count = len(all_logs[all_logs['model'] == model])
        unique_configs = all_logs[all_logs['model'] == model]['filename'].nunique()
        print(f"  - {model}: {model_count}条记录, {unique_configs}个配置")
        
        # 查找每个模型的dropout和weight_decay组合
        model_params = all_logs[all_logs['model'] == model][['dropout', 'weight_decay', 'filename']].drop_duplicates()
        print(f"    {model}的参数配置:")
        for _, params in model_params.iterrows():
            print(f"      dropout={params['dropout']}, weight_decay={params['weight_decay']}, 文件={params['filename']}")
    
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
            
            model_summary.append(summary_dict)
    
    summary_df = pd.DataFrame(model_summary)
    summary_df.to_csv(os.path.join(output_dir, "model_performance_summary.csv"), index=False)
    print(f"模型性能摘要已保存至 {os.path.join(output_dir, 'model_performance_summary.csv')}")
    
    # 2. 为每个指标绘制对比图 - 专注于dropout和weight_decay比较
    # 2.1 首先按模型类型绘制不同dp值的比较图
    for model in models:
        model_data = all_logs[all_logs['model'] == model]
        unique_dropouts = model_data['dropout'].unique()
        
        for metric in metrics:
            plt.figure(figsize=(14, 8))
            
            for i, dp in enumerate(unique_dropouts):
                dp_data = model_data[model_data['dropout'] == dp]
                # 取这个dp值下性能最好的weight_decay配置
                best_wd_config = dp_data.loc[dp_data.groupby('filename')['val_acc'].idxmax()]
                
                if not best_wd_config.empty:
                    best_filename = best_wd_config['filename'].iloc[0]
                    best_config_data = model_data[(model_data['dropout'] == dp) & 
                                                 (model_data['filename'] == best_filename)]
                    best_config_data = best_config_data.sort_values('epoch')
                    
                    wd = best_config_data['weight_decay'].iloc[0]
                    label = f"dp={dp:.2f}, wd={wd:.6f}"
                    
                    plt.plot(best_config_data['epoch'], best_config_data[metric], 
                             label=label, marker='o', color=colors[i % len(colors)], linewidth=2)
            
            plt.title(f'{model} - {metric} (不同dropout值对比)', fontsize=15, fontproperties=songti_font)
            plt.xlabel('Epoch', fontsize=12, fontproperties=songti_font)
            plt.ylabel(metric, fontsize=12, fontproperties=songti_font)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(loc='best', fontsize=11, prop=songti_font)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model}_{metric}_dp_comparison.png"), dpi=300)
            plt.close()
            
        print(f"已生成 {model} 模型的dropout对比图")
    
    # 2.2 按模型类型绘制不同wd值的比较图
    for model in models:
        model_data = all_logs[all_logs['model'] == model]
        unique_wds = model_data['weight_decay'].unique()
        
        for metric in metrics:
            plt.figure(figsize=(14, 8))
            
            for i, wd in enumerate(unique_wds):
                wd_data = model_data[model_data['weight_decay'] == wd]
                # 取这个wd值下性能最好的dropout配置
                best_dp_config = wd_data.loc[wd_data.groupby('filename')['val_acc'].idxmax()]
                
                if not best_dp_config.empty:
                    best_filename = best_dp_config['filename'].iloc[0]
                    best_config_data = model_data[(model_data['weight_decay'] == wd) & 
                                                 (model_data['filename'] == best_filename)]
                    best_config_data = best_config_data.sort_values('epoch')
                    
                    dp = best_config_data['dropout'].iloc[0]
                    label = f"dp={dp:.2f}, wd={wd:.6f}"
                    
                    plt.plot(best_config_data['epoch'], best_config_data[metric], 
                             label=label, marker='o', color=colors[i % len(colors)], linewidth=2)
            
            plt.title(f'{model} - {metric} (不同weight_decay值对比)', fontsize=15, fontproperties=songti_font)
            plt.xlabel('Epoch', fontsize=12, fontproperties=songti_font)
            plt.ylabel(metric, fontsize=12, fontproperties=songti_font)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(loc='best', fontsize=11, prop=songti_font)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model}_{metric}_wd_comparison.png"), dpi=300)
            plt.close()
            
        print(f"已生成 {model} 模型的weight_decay对比图")
    
    # 2.3 创建不同模型最佳配置的比较图
    for metric in metrics:
        plt.figure(figsize=(16, 10))
        
        # 获取每个模型的最佳配置
        best_configs = summary_df.loc[summary_df.groupby('model')['val_acc'].idxmax()]
        
        for i, (idx, row) in enumerate(best_configs.iterrows()):
            model = row['model']
            filename = row['filename']
            
            # 获取该配置的完整数据
            config_data = all_logs[(all_logs['model'] == model) & (all_logs['filename'] == filename)]
            config_data = config_data.sort_values('epoch')
            
            # 简化标签：只显示dp和wd
            label = f"{model} (dp={row['dropout']:.2f}, wd={row['weight_decay']:.6f})"
            
            plt.plot(config_data['epoch'], config_data[metric], 
                     label=label, marker='o', color=colors[i % len(colors)], linewidth=2)
        
        plt.title(f'不同模型最佳配置 {metric} 对比', fontsize=15, fontproperties=songti_font)
        plt.xlabel('Epoch', fontsize=12, fontproperties=songti_font)
        plt.ylabel(metric, fontsize=12, fontproperties=songti_font)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(loc='best', fontsize=12, prop=songti_font)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_best_config_comparison.png"), dpi=300)
        plt.close()
        
    print(f"已生成最佳配置对比图")
    
    # 3. 专注于生成模型的dp vs wd热力图
    for model in models:
        model_specific = summary_df[summary_df['model'] == model]
        print(f"处理{model}的热力图, 有{len(model_specific)}个数据点")
        
        # 为每个模型创建dropout vs weight_decay热力图
        plt.figure(figsize=(14, 10))
        try:
            pivot_data = pd.pivot_table(
                model_specific,
                values='val_acc',
                index='dropout',
                columns='weight_decay',
                aggfunc='mean'
            )
            if not pivot_data.empty and pivot_data.shape[0] > 0 and pivot_data.shape[1] > 0:
                sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.2f', annot_kws={"size": 12})
                plt.title(f'{model} 验证准确率: Dropout vs Weight Decay', fontsize=16, fontproperties=songti_font)
                plt.xlabel('Weight Decay', fontsize=14, fontproperties=songti_font)
                plt.ylabel('Dropout', fontsize=14, fontproperties=songti_font)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{model}_dropout_wd_heatmap.png"), dpi=300)
                print(f"已生成 {model} dropout vs weight_decay 热力图")
            else:
                print(f"{model} 数据不足以生成 dropout vs weight_decay 热力图")
        except Exception as e:
            print(f"生成 {model} dropout vs weight_decay 热力图时出错: {e}")
        finally:
            plt.close()
    
    # 4. 绘制所有模型的性能柱状图比较
    best_model_data = summary_df.sort_values('val_acc', ascending=False)
    
    # 创建子图，展示val_acc, train_acc, f1和recall
    plt.figure(figsize=(18, 10))
    metrics_to_plot = ['val_acc', 'train_acc', 'f1', 'recall']
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        ax = sns.barplot(x='model', y=metric, hue='dropout', data=best_model_data, palette='viridis')
        plt.title(f'最佳 {metric} (按模型和dropout)', fontsize=14, fontproperties=songti_font)
        plt.xticks(rotation=45, fontsize=12, fontproperties=songti_font)
        plt.yticks(fontsize=12, fontproperties=songti_font)
        # 在条形上添加数值标签
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom',
                        fontsize=10, fontproperties=songti_font)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_metrics_by_dropout.png"), dpi=300)
    plt.close()
    
    # 使用weight_decay作为分组绘制柱状图
    plt.figure(figsize=(18, 10))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        ax = sns.barplot(x='model', y=metric, hue='weight_decay', data=best_model_data, palette='plasma')
        plt.title(f'最佳 {metric} (按模型和weight_decay)', fontsize=14, fontproperties=songti_font)
        plt.xticks(rotation=45, fontsize=12, fontproperties=songti_font)
        plt.yticks(fontsize=12, fontproperties=songti_font)
        # 在条形上添加数值标签
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'bottom',
                        fontsize=9, fontproperties=songti_font)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_metrics_by_weight_decay.png"), dpi=300)
    plt.close()
    
    print(f"已生成按dropout和weight_decay分组的性能比较图")

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
    <title>模型训练性能对比分析 - Dropout和Weight Decay参数研究</title>
    <style>
        body {{ font-family: "Songti SC", "宋体", serif; margin: 20px; }}
        h1, h2, h3, h4, h5, h6 {{ color: #2c3e50; font-family: "Songti SC", "宋体", serif; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; font-family: "Songti SC", "宋体", serif; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        p, li, ul, ol {{ font-family: "Songti SC", "宋体", serif; }}
        .metric-img {{ max-width: 100%; margin-top: 10px; }}
        .container {{ display: flex; flex-wrap: wrap; }}
        .chart {{ margin: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 15px; }}
        .highlight {{ background-color: #e8f4f8; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>模型训练性能对比分析报告 - Dropout和Weight Decay参数研究</h1>
    
    <h2>最佳性能模型</h2>
    <p>整体最佳模型: <strong>{best_overall['model']} (验证准确率: {best_overall['val_acc']:.2f}%)</strong></p>
    <p>最佳参数配置: dropout={best_overall['dropout']:.2f}, weight_decay={best_overall['weight_decay']:.6f}</p>
    <p>最佳epoch: {best_overall['best_epoch']}</p>
    """
    
    html_content += """
    <h2>各模型最佳配置</h2>
    <table>
        <tr>
            <th>模型</th>
            <th>Dropout</th>
            <th>Weight Decay</th>
            <th>验证准确率</th>
            <th>训练准确率</th>
            <th>F1分数</th>
            <th>召回率</th>
            <th>最佳Epoch</th>
        </tr>
    """
    
    for _, row in best_configs.iterrows():
        html_content += f"""
        <tr class="{'highlight' if row['model'] == best_overall['model'] else ''}">
            <td>{row['model']}</td>
            <td>{row['dropout']}</td>
            <td>{row['weight_decay']:.6f}</td>
            <td>{row['val_acc']:.2f}%</td>
            <td>{row['train_acc']:.2f}%</td>
            <td>{row['f1']:.2f}%</td>
            <td>{row['recall']:.2f}%</td>
            <td>{row['best_epoch']}</td>
        </tr>
        """
    
    html_content += """
    </table>
    
    <h2>不同Dropout和Weight Decay下的性能比较</h2>
    <div class="container">
    """
    
    # 添加dp vs wd热力图
    heatmap_files = [f for f in os.listdir(output_dir) if "dropout_wd_heatmap" in f and f.endswith(".png")]
    for heatmap_file in heatmap_files:
        model_name = heatmap_file.split("_dropout_wd_heatmap")[0]
        html_content += f"""
        <div class="chart" style="width: 48%;">
            <h3>{model_name} Dropout vs Weight Decay 热力图</h3>
            <img src="{heatmap_file}" alt="{model_name} Dropout vs Weight Decay Heatmap" class="metric-img">
        </div>
        """
    
    html_content += """
    </div>
    
    <h2>模型性能指标对比</h2>
    <div class="container">
    """
    
    # 添加按dropout和weight_decay分组的性能比较图
    metrics_comparison_files = ["model_metrics_by_dropout.png", "model_metrics_by_weight_decay.png"]
    for file in metrics_comparison_files:
        if os.path.exists(os.path.join(output_dir, file)):
            html_content += f"""
            <div class="chart" style="width: 95%;">
                <h3>{"按Dropout分组" if "dropout" in file else "按Weight Decay分组"}的模型性能比较</h3>
                <img src="{file}" alt="Model metrics comparison" class="metric-img">
            </div>
            """
    
    html_content += """
    </div>
    
    <h2>各模型最佳配置的指标对比</h2>
    <div class="container">
    """
    
    # 添加各指标的最佳配置对比图
    metrics = ['val_acc', 'train_acc', 'val_loss', 'train_loss', 'f1', 'recall']
    for metric in metrics:
        best_config_file = f"{metric}_best_config_comparison.png"
        if os.path.exists(os.path.join(output_dir, best_config_file)):
            html_content += f"""
            <div class="chart" style="width: 95%;">
                <h3>{metric} 最佳配置对比</h3>
                <img src="{best_config_file}" alt="{metric} best config comparison" class="metric-img">
            </div>
            """
    
    html_content += """
    </div>
    
    <h2>各模型不同Dropout值的性能比较</h2>
    """
    
    # 为每个模型添加dropout比较图
    models = best_configs['model'].unique()
    for model in models:
        html_content += f"""
        <h3>{model} 模型不同Dropout值的性能比较</h3>
        <div class="container">
        """
        
        for metric in metrics[:4]:  # 只显示前4个关键指标
            dp_comparison_file = f"{model}_{metric}_dp_comparison.png"
            if os.path.exists(os.path.join(output_dir, dp_comparison_file)):
                html_content += f"""
                <div class="chart" style="width: 48%;">
                    <h4>{metric} (不同dropout值)</h4>
                    <img src="{dp_comparison_file}" alt="{model} {metric} dropout comparison" class="metric-img">
                </div>
                """
        
        html_content += """
        </div>
        """
    
    html_content += """
    <h2>各模型不同Weight Decay值的性能比较</h2>
    """
    
    # 为每个模型添加weight_decay比较图
    for model in models:
        html_content += f"""
        <h3>{model} 模型不同Weight Decay值的性能比较</h3>
        <div class="container">
        """
        
        for metric in metrics[:4]:  # 只显示前4个关键指标
            wd_comparison_file = f"{model}_{metric}_wd_comparison.png"
            if os.path.exists(os.path.join(output_dir, wd_comparison_file)):
                html_content += f"""
                <div class="chart" style="width: 48%;">
                    <h4>{metric} (不同weight_decay值)</h4>
                    <img src="{wd_comparison_file}" alt="{model} {metric} weight_decay comparison" class="metric-img">
                </div>
                """
        
        html_content += """
        </div>
        """
    
    html_content += """
    <h2>结论与建议</h2>
    <p>根据对不同模型、不同dropout和weight_decay参数配置的性能分析，我们得出以下结论：</p>
    <ul>
    """
    
    # 添加结论
    best_overall_model = best_overall['model']
    best_dropout = best_overall['dropout']
    best_weight_decay = best_overall['weight_decay']
    html_content += f"""
        <li>整体性能最佳的模型是 <strong>{best_overall_model}</strong>，验证准确率达到 {best_overall['val_acc']:.2f}%</li>
        <li>该模型的最佳参数配置是：dropout={best_dropout:.2f}, weight_decay={best_weight_decay:.6f}</li>
    """
    
    # 为每个模型添加最佳参数建议
    html_content += "<li>各模型的最佳参数配置：</li><ul>"
    for _, row in best_configs.iterrows():
        html_content += f"""
            <li>{row['model']}: dropout={row['dropout']:.2f}, weight_decay={row['weight_decay']:.6f}</li>
        """
    html_content += "</ul>"
    
    # 添加一般性观察
    html_content += """
        <li>通过热力图可以观察到，dropout和weight_decay的不同组合对模型性能有显著影响</li>
        <li>较小的weight_decay值（如1e-5）通常能够获得更好的验证准确率</li>
        <li>过高或过低的dropout值都可能导致模型性能下降</li>
    </ul>
    
    <footer>
        <p>此报告由模型对比分析工具自动生成，专注于Dropout和Weight Decay参数的影响分析</p>
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
