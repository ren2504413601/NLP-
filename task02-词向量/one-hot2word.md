
# one-hot

### 词袋模型(BOW,bag of words)
词袋模型是自然语言处理中在建模文本时常用的文本表示方法。

词袋模型是在自然语言处理和信息检索中的一种简单假设。在这种模型中，文本（段落或者文档）被看作是无序的词汇集合，忽略语法甚至是单词的顺序。

把句子转换成一个稀疏向量。

规则是：对应索引位置上的单词存在，则对应索引值是1


### TF-IDF（term frequency–inverse document frequency）

主要思想是：如果某个词或短语在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。TFIDF实际上是：TF * IDF，TF词频(Term Frequency)，IDF逆向文件频率(Inverse Document Frequency)。TF表示词条在文档d中出现的频率。IDF的主要思想是：如果包含词条t的文档越少，也就是n越小，IDF越大，则说明词条t具有很好的类别区分能力。

计算公式：
$$ w_{tf-idf}=w_{tf} \cdot \log {\frac{1}{w_{df}} } $$
其中，$w_{tf}$是文档的词频，$ w_{df} $是包含该单词的所有文档的总频率

**参考**：https://baike.baidu.com/item/tf-idf/8816134?fr=aladdin

# word2vec/word embedding/word representations

**Idea:**

• We have a large corpus of text

• Every word in a fixed vocabulary is represented by a vector

• Go through each position t in the text, which has a center word c and context (“outside”) words o

• Use the similarity of the word vectors for c and o to calculate the probability of o given c (or vice versa)

• Keep adjusting the word vectors to maximize this probability


基本思想：创建词向量来体现单词的上下文关系，给出一个上下文相关的单词集合来预测目标单词

**(a)连续词带管理(CBOW)**

给定上下文预测中心的单词

**(b)skip-gram模型**

给定中心单词预测上下文

**参考**：http://www.hankcs.com/nlp/word2vec.html
