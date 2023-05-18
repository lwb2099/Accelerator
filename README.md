# 编程训练
from 2023.3 - 2023.7
### 说明：

下列实验训练数据无特殊说明的话超过100k的只截取前100k，不要改动数据集顺序，输入token长度超过256的限制为256即可。
要求给出baseline和改进后模型在测试集上的结果，除特殊说明外评估指标采用和给定论文一致。
数据可采用如下数据链接进行下载，也可以自己通过别的链接下载。
每个任务时间限制为3周，单人完成，需要提交文档说明。
### 1.多卡：
SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization
测试数据集采用CNN/DM
数据链接：cnn_dailymail · Datasets at Hugging Face
将该评估指标从单卡改成多卡（建议采用Accelerate库），
注意：这里多卡我们统一使用2卡进行测试。
使用CNN/DM测试集进行测试，多卡评估结果应当和单卡评估结果基本一致。

### 2.预训练：
BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
数据链接：bookcorpus · Datasets at Hugging Face
选择前面100k作为训练数据，接着的1k数据作为验证，再之后1k用于测试。
评估指标为loss
使用Transformers库实现Bart预训练中的Text Infilling，可基于Transformers库中现有的Examples

### 3.相对位置编码：
Self-Attention with Relative Position Representations
训练数据集采用IWSLT En-De
测试数据集采用IWSLT En-De
修改fairseq中任意transformer的代码（建议使用fairseq中iwslt对应的模型）以实现相对论文中的相对位置编码
数据链接：fairseq/examples/translation at main · facebookresearch/fairseq (github.com)
### 4.对抗训练：
Towards Robust Neural Machine Translation
训练数据集采用IWSLT En-Fr
测试数据集采用newstest2014 test set En-Fr
数据链接：fairseq/examples/translation at main · facebookresearch/fairseq (github.com)
实现该论文里面的对抗训练方法

### 5.对比学习：
SimCSE: Simple Contrastive Learning of Sentence Embeddings
训练及测试数据集使用MNLI
数据链接：SetFit/mnli · Datasets at Hugging Face
实现论文中Supervised model

### 6.强化学习：
A Study of Reinforcement Learning for Neural Machine Translation
训练数据集采用IWSLT En-Fr
测试数据集采用newstest2014 test set En-Fr
数据链接：fairseq/examples/translation at main · facebookresearch/fairseq (github.com)
实现论文Table 1中RL (multinomial + terminal) 模型
