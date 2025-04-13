以下是 Word2Vec 中 Skip-gram 和 CBOW 模型的数学公式解释及 LaTeX 表达：

---

### **1. Word2Vec 基础概念**
Word2Vec 的核心是通过预测上下文词（Skip-gram）或根据上下文预测目标词（CBOW），学习词的稠密向量表示（Embedding）。  
- **符号定义**：
  - \( V \): 词汇表大小（总词数）
  - \( d \): 词向量维度（如 300）
  - \( w_t \): 时间步 \( t \) 的目标词（中心词）
  - \( w_{t+j} \): 上下文词（\( j \) 为上下文窗口偏移量）
  - \( W \in \mathbb{R}^{V \times d} \): 输入词向量矩阵（每个词的 one-hot 向量映射到 \( d \) 维向量）
  - \( W' \in \mathbb{R}^{d \times V} \): 输出权重矩阵（将词向量映射回词汇表空间）

---

### **2. Skip-gram 模型**
**目标**：给定中心词 \( w_t \)，预测其上下文词 \( w_{t+j} \)。

#### **公式推导**
1. **输入层**：  
   输入为中心词的 one-hot 向量 \( x \in \mathbb{R}^V \)。
2. **隐藏层**：  
   通过输入词向量矩阵 \( W \) 得到中心词的嵌入向量：  
   \[
   h = W^T x
   \]
3. **输出层**：  
   通过输出矩阵 \( W' \) 计算上下文词的得分，并用 softmax 归一化为概率：  
   \[
   p(w_{t+j} | w_t) = \frac{\exp(h^T W'_{\cdot, w_{t+j}})}{\sum_{v=1}^V \exp(h^T W'_{\cdot, v})}
   \]
   其中 \( W'_{\cdot, v} \) 表示矩阵 \( W' \) 的第 \( v \) 列。

4. **目标函数**：  
   最大化对数似然：  
   \[
   \mathcal{L}_{\text{skip-gram}} = \sum_{t=1}^T \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)
   \]
   其中 \( c \) 是上下文窗口大小。

---

### **3. CBOW 模型**
**目标**：给定上下文词 \( w_{t-1}, w_{t+1} \)，预测中心词 \( w_t \)。

#### **公式推导**
1. **输入层**：  
   输入为上下文词的 one-hot 向量集合 \( x_{t-1}, x_{t+1} \)。
2. **隐藏层**：  
   对上下文词的嵌入向量取平均：  
   \[
   h = \frac{1}{2c} \sum_{j=-c}^{c} W^T x_{t+j} \quad (j \neq 0)
   \]
3. **输出层**：  
   通过 softmax 预测中心词的概率：  
   \[
   p(w_t | w_{\text{context}}) = \frac{\exp(h^T W'_{\cdot, w_t})}{\sum_{v=1}^V \exp(h^T W'_{\cdot, v})}
   \]
4. **目标函数**：  
   最大化对数似然：  
   \[
   \mathcal{L}_{\text{CBOW}} = \sum_{t=1}^T \log p(w_t | w_{\text{context}})
   \]

---

### **4. 优化技巧**
由于标准 softmax 计算复杂度为 \( O(V) \)，实际训练中常用以下优化：
1. **负采样（Negative Sampling）**：  
   仅更新目标词和少数负样本的参数，替代全 softmax。
2. **层次 Softmax（Hierarchical Softmax）**：  
   利用二叉树结构将复杂度降为 \( O(\log V) \)。

---

### **公式总结**
| 模型      | 输入到隐藏层                      | 隐藏层到输出层 | 目标函数                                        |
| --------- | --------------------------------- | -------------- | ----------------------------------------------- |
| Skip-gram | \( h = W^T x \)                   | \( p(w_{t+j}   | w_t) = \text{softmax}(h^T W') \)                |
| CBOW      | \( h = \frac{1}{2c} \sum W^T x \) | \( p(w_t       | w_{\text{context}}) = \text{softmax}(h^T W') \) |

---

### **LaTeX 代码**
```latex
% Skip-gram 公式
p(w_{t+j} | w_t) = \frac{\exp(h^T W'_{\cdot, w_{t+j}})}{\sum_{v=1}^V \exp(h^T W'_{\cdot, v})}

% CBOW 公式
p(w_t | w_{\text{context}}) = \frac{\exp(h^T W'_{\cdot, w_t})}{\sum_{v=1}^V \exp(h^T W'_{\cdot, v})}
```

通过上述公式，Skip-gram 和 CBOW 分别通过不同方向的预测任务，将词的语义关系编码到低维向量空间中。
