
# transformer

首先将这个模型看成是一个黑箱操作。在机器翻译中，就是输入一种语言，输出另一种语言。

那么拆开这个黑箱，我们可以看到它是由编码组件、解码组件和它们之间的连接组成。

编码组件部分由一堆编码器（encoder）构成（论文中是将6个编码器叠在一起——数字6没有什么神奇之处，你也可以尝试其他数字）。解码组件部分也是由相同数量（与编码器对应）的解码器（decoder）组成的。
![](transformer.jpg)
所有的编码器在结构上都是相同的，但它们没有共享参数。每个解码器都可以分解成两个子层。

![](self-attention.jpg)
从编码器输入的句子首先会经过一个自注意力（self-attention）层，这层帮助编码器在对每个单词编码时关注输入句子的其他单词。我们将在稍后的文章中更深入地研究自注意力。

自注意力层的输出会传递到前馈（feed-forward）神经网络中。每个位置的单词对应的前馈神经网络都完全一样（译注：另一种解读就是一层窗口为一个单词的一维卷积神经网络）。

解码器中也有编码器的自注意力（self-attention）层和前馈（feed-forward）层。除此之外，这两个层之间还有一个注意力层，用来关注输入句子的相关部分（和seq2seq模型的注意力作用相似）。

![](transformer-enc-dec.png)
**在 Encoder 中，**

1.Input 经过 embedding 后，要做 positional encodings，

2.然后是 Multi-head attention，

3.再经过 position-wise Feed Forward，

4.每个子层之间有残差连接。

**在 Decoder 中，**

1.如上图所示，也有 positional encodings，Multi-head attention 和 FFN，子层之间也要做残差连接，

2.但比 encoder 多了一个 Masked Multi-head attention，

3.最后要经过 Linear 和 softmax 输出概率。



**参考：**https://baijiahao.baidu.com/s?id=1622064575970777188&wfr=spider&for=pc

### 以tensor2tensor 的hello_t2t.ipynb 文件为例

[hello_t2t.ipynb](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)


```python
# Imports we need.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import collections

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

# Enable TF Eager execution
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys

# Setup some directories
data_dir = os.path.expanduser("~/t2t/data")
tmp_dir = os.path.expanduser("~/t2t/tmp")
train_dir = os.path.expanduser("~/t2t/train")
checkpoint_dir = os.path.expanduser("~/t2t/checkpoints")
tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)
tf.gfile.MakeDirs(train_dir)
tf.gfile.MakeDirs(checkpoint_dir)
gs_data_dir = "gs://tensor2tensor-data"
gs_ckpt_dir = "gs://tensor2tensor-checkpoints/"
```

    WARNING: Logging before flag parsing goes to stderr.
    W0822 15:57:25.659379  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\utils\expert_utils.py:68: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.
    
    W0822 15:57:51.742776  9748 lazy_loader.py:50] 
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
      * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.
    
    W0822 15:57:55.109573  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\utils\adafactor.py:27: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    W0822 15:57:55.169566  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\utils\multistep_optimizer.py:32: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.
    
    W0822 15:57:58.149388  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\mesh_tensorflow\ops.py:4237: The name tf.train.CheckpointSaverListener is deprecated. Please use tf.estimator.CheckpointSaverListener instead.
    
    W0822 15:57:58.151390  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\mesh_tensorflow\ops.py:4260: The name tf.train.SessionRunHook is deprecated. Please use tf.estimator.SessionRunHook instead.
    
    W0822 15:57:59.202363  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\rl\gym_utils.py:219: The name tf.logging.info is deprecated. Please use tf.compat.v1.logging.info instead.
    
    W0822 15:58:01.824192  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\utils\trainer_lib.py:109: The name tf.OptimizerOptions is deprecated. Please use tf.compat.v1.OptimizerOptions instead.
    
    


```python

# A Problem is a dataset together with some fixed pre-processing.
# It could be a translation dataset with a specific tokenization,
# or an image dataset with a specific resolution.
#
# There are many problems available in Tensor2Tensor
problems.available()
```




    ['algorithmic_addition_binary40',
     'algorithmic_addition_decimal40',
     'algorithmic_cipher_shift200',
     'algorithmic_cipher_shift5',
     'algorithmic_cipher_vigenere200',
     'algorithmic_cipher_vigenere5',
     'algorithmic_identity_binary40',
     'algorithmic_identity_decimal40',
     'algorithmic_math_deepmind_all',
     'algorithmic_math_two_variables',
     'algorithmic_multiplication_binary40',
     'algorithmic_multiplication_decimal40',
     'algorithmic_reverse_binary40',
     'algorithmic_reverse_binary40_test',
     'algorithmic_reverse_decimal40',
     'algorithmic_reverse_nlplike32k',
     'algorithmic_reverse_nlplike8k',
     'algorithmic_shift_decimal40',
     'algorithmic_sort_problem',
     'audio_timit_characters_tune',
     'audio_timit_tokens8k_test',
     'audio_timit_tokens8k_tune',
     'babi_qa_concat_all_tasks_10k',
     'babi_qa_concat_all_tasks_1k',
     'babi_qa_concat_task10_10k',
     'babi_qa_concat_task10_1k',
     'babi_qa_concat_task11_10k',
     'babi_qa_concat_task11_1k',
     'babi_qa_concat_task12_10k',
     'babi_qa_concat_task12_1k',
     'babi_qa_concat_task13_10k',
     'babi_qa_concat_task13_1k',
     'babi_qa_concat_task14_10k',
     'babi_qa_concat_task14_1k',
     'babi_qa_concat_task15_10k',
     'babi_qa_concat_task15_1k',
     'babi_qa_concat_task16_10k',
     'babi_qa_concat_task16_1k',
     'babi_qa_concat_task17_10k',
     'babi_qa_concat_task17_1k',
     'babi_qa_concat_task18_10k',
     'babi_qa_concat_task18_1k',
     'babi_qa_concat_task19_10k',
     'babi_qa_concat_task19_1k',
     'babi_qa_concat_task1_10k',
     'babi_qa_concat_task1_1k',
     'babi_qa_concat_task20_10k',
     'babi_qa_concat_task20_1k',
     'babi_qa_concat_task2_10k',
     'babi_qa_concat_task2_1k',
     'babi_qa_concat_task3_10k',
     'babi_qa_concat_task3_1k',
     'babi_qa_concat_task4_10k',
     'babi_qa_concat_task4_1k',
     'babi_qa_concat_task5_10k',
     'babi_qa_concat_task5_1k',
     'babi_qa_concat_task6_10k',
     'babi_qa_concat_task6_1k',
     'babi_qa_concat_task7_10k',
     'babi_qa_concat_task7_1k',
     'babi_qa_concat_task8_10k',
     'babi_qa_concat_task8_1k',
     'babi_qa_concat_task9_10k',
     'babi_qa_concat_task9_1k',
     'cola',
     'cola_characters',
     'common_voice',
     'common_voice_clean',
     'common_voice_noisy',
     'common_voice_train_full_test_clean',
     'copy_sequence',
     'copy_sequence_small',
     'flip_bi_gram_sequence',
     'genomics_expression_cage10',
     'genomics_expression_gm12878',
     'genomics_expression_l262k',
     'github_function_docstring',
     'gym_air_raid-v0_random',
     'gym_air_raid-v4_random',
     'gym_air_raid_deterministic-v0_random',
     'gym_air_raid_deterministic-v4_random',
     'gym_air_raid_no_frameskip-v0_random',
     'gym_air_raid_no_frameskip-v4_random',
     'gym_alien-v0_random',
     'gym_alien-v4_random',
     'gym_alien_deterministic-v0_random',
     'gym_alien_deterministic-v4_random',
     'gym_alien_no_frameskip-v0_random',
     'gym_alien_no_frameskip-v4_random',
     'gym_amidar-v0_random',
     'gym_amidar-v4_random',
     'gym_amidar_deterministic-v0_random',
     'gym_amidar_deterministic-v4_random',
     'gym_amidar_no_frameskip-v0_random',
     'gym_amidar_no_frameskip-v4_random',
     'gym_assault-v0_random',
     'gym_assault-v4_random',
     'gym_assault_deterministic-v0_random',
     'gym_assault_deterministic-v4_random',
     'gym_assault_no_frameskip-v0_random',
     'gym_assault_no_frameskip-v4_random',
     'gym_asterix-v0_random',
     'gym_asterix-v4_random',
     'gym_asterix_deterministic-v0_random',
     'gym_asterix_deterministic-v4_random',
     'gym_asterix_no_frameskip-v0_random',
     'gym_asterix_no_frameskip-v4_random',
     'gym_asteroids-v0_random',
     'gym_asteroids-v4_random',
     'gym_asteroids_deterministic-v0_random',
     'gym_asteroids_deterministic-v4_random',
     'gym_asteroids_no_frameskip-v0_random',
     'gym_asteroids_no_frameskip-v4_random',
     'gym_atlantis-v0_random',
     'gym_atlantis-v4_random',
     'gym_atlantis_deterministic-v0_random',
     'gym_atlantis_deterministic-v4_random',
     'gym_atlantis_no_frameskip-v0_random',
     'gym_atlantis_no_frameskip-v4_random',
     'gym_bank_heist-v0_random',
     'gym_bank_heist-v4_random',
     'gym_bank_heist_deterministic-v0_random',
     'gym_bank_heist_deterministic-v4_random',
     'gym_bank_heist_no_frameskip-v0_random',
     'gym_bank_heist_no_frameskip-v4_random',
     'gym_battle_zone-v0_random',
     'gym_battle_zone-v4_random',
     'gym_battle_zone_deterministic-v0_random',
     'gym_battle_zone_deterministic-v4_random',
     'gym_battle_zone_no_frameskip-v0_random',
     'gym_battle_zone_no_frameskip-v4_random',
     'gym_beam_rider-v0_random',
     'gym_beam_rider-v4_random',
     'gym_beam_rider_deterministic-v0_random',
     'gym_beam_rider_deterministic-v4_random',
     'gym_beam_rider_no_frameskip-v0_random',
     'gym_beam_rider_no_frameskip-v4_random',
     'gym_berzerk-v0_random',
     'gym_berzerk-v4_random',
     'gym_berzerk_deterministic-v0_random',
     'gym_berzerk_deterministic-v4_random',
     'gym_berzerk_no_frameskip-v0_random',
     'gym_berzerk_no_frameskip-v4_random',
     'gym_bowling-v0_random',
     'gym_bowling-v4_random',
     'gym_bowling_deterministic-v0_random',
     'gym_bowling_deterministic-v4_random',
     'gym_bowling_no_frameskip-v0_random',
     'gym_bowling_no_frameskip-v4_random',
     'gym_boxing-v0_random',
     'gym_boxing-v4_random',
     'gym_boxing_deterministic-v0_random',
     'gym_boxing_deterministic-v4_random',
     'gym_boxing_no_frameskip-v0_random',
     'gym_boxing_no_frameskip-v4_random',
     'gym_breakout-v0_random',
     'gym_breakout-v4_random',
     'gym_breakout_deterministic-v0_random',
     'gym_breakout_deterministic-v4_random',
     'gym_breakout_no_frameskip-v0_random',
     'gym_breakout_no_frameskip-v4_random',
     'gym_carnival-v0_random',
     'gym_carnival-v4_random',
     'gym_carnival_deterministic-v0_random',
     'gym_carnival_deterministic-v4_random',
     'gym_carnival_no_frameskip-v0_random',
     'gym_carnival_no_frameskip-v4_random',
     'gym_centipede-v0_random',
     'gym_centipede-v4_random',
     'gym_centipede_deterministic-v0_random',
     'gym_centipede_deterministic-v4_random',
     'gym_centipede_no_frameskip-v0_random',
     'gym_centipede_no_frameskip-v4_random',
     'gym_chopper_command-v0_random',
     'gym_chopper_command-v4_random',
     'gym_chopper_command_deterministic-v0_random',
     'gym_chopper_command_deterministic-v4_random',
     'gym_chopper_command_no_frameskip-v0_random',
     'gym_chopper_command_no_frameskip-v4_random',
     'gym_crazy_climber-v0_random',
     'gym_crazy_climber-v4_random',
     'gym_crazy_climber_deterministic-v0_random',
     'gym_crazy_climber_deterministic-v4_random',
     'gym_crazy_climber_no_frameskip-v0_random',
     'gym_crazy_climber_no_frameskip-v4_random',
     'gym_demon_attack-v0_random',
     'gym_demon_attack-v4_random',
     'gym_demon_attack_deterministic-v0_random',
     'gym_demon_attack_deterministic-v4_random',
     'gym_demon_attack_no_frameskip-v0_random',
     'gym_demon_attack_no_frameskip-v4_random',
     'gym_double_dunk-v0_random',
     'gym_double_dunk-v4_random',
     'gym_double_dunk_deterministic-v0_random',
     'gym_double_dunk_deterministic-v4_random',
     'gym_double_dunk_no_frameskip-v0_random',
     'gym_double_dunk_no_frameskip-v4_random',
     'gym_elevator_action-v0_random',
     'gym_elevator_action-v4_random',
     'gym_elevator_action_deterministic-v0_random',
     'gym_elevator_action_deterministic-v4_random',
     'gym_elevator_action_no_frameskip-v0_random',
     'gym_elevator_action_no_frameskip-v4_random',
     'gym_enduro-v0_random',
     'gym_enduro-v4_random',
     'gym_enduro_deterministic-v0_random',
     'gym_enduro_deterministic-v4_random',
     'gym_enduro_no_frameskip-v0_random',
     'gym_enduro_no_frameskip-v4_random',
     'gym_fishing_derby-v0_random',
     'gym_fishing_derby-v4_random',
     'gym_fishing_derby_deterministic-v0_random',
     'gym_fishing_derby_deterministic-v4_random',
     'gym_fishing_derby_no_frameskip-v0_random',
     'gym_fishing_derby_no_frameskip-v4_random',
     'gym_freeway-v0_random',
     'gym_freeway-v4_random',
     'gym_freeway_deterministic-v0_random',
     'gym_freeway_deterministic-v4_random',
     'gym_freeway_no_frameskip-v0_random',
     'gym_freeway_no_frameskip-v4_random',
     'gym_frostbite-v0_random',
     'gym_frostbite-v4_random',
     'gym_frostbite_deterministic-v0_random',
     'gym_frostbite_deterministic-v4_random',
     'gym_frostbite_no_frameskip-v0_random',
     'gym_frostbite_no_frameskip-v4_random',
     'gym_gopher-v0_random',
     'gym_gopher-v4_random',
     'gym_gopher_deterministic-v0_random',
     'gym_gopher_deterministic-v4_random',
     'gym_gopher_no_frameskip-v0_random',
     'gym_gopher_no_frameskip-v4_random',
     'gym_gravitar-v0_random',
     'gym_gravitar-v4_random',
     'gym_gravitar_deterministic-v0_random',
     'gym_gravitar_deterministic-v4_random',
     'gym_gravitar_no_frameskip-v0_random',
     'gym_gravitar_no_frameskip-v4_random',
     'gym_hero-v0_random',
     'gym_hero-v4_random',
     'gym_hero_deterministic-v0_random',
     'gym_hero_deterministic-v4_random',
     'gym_hero_no_frameskip-v0_random',
     'gym_hero_no_frameskip-v4_random',
     'gym_ice_hockey-v0_random',
     'gym_ice_hockey-v4_random',
     'gym_ice_hockey_deterministic-v0_random',
     'gym_ice_hockey_deterministic-v4_random',
     'gym_ice_hockey_no_frameskip-v0_random',
     'gym_ice_hockey_no_frameskip-v4_random',
     'gym_jamesbond-v0_random',
     'gym_jamesbond-v4_random',
     'gym_jamesbond_deterministic-v0_random',
     'gym_jamesbond_deterministic-v4_random',
     'gym_jamesbond_no_frameskip-v0_random',
     'gym_jamesbond_no_frameskip-v4_random',
     'gym_journey_escape-v0_random',
     'gym_journey_escape-v4_random',
     'gym_journey_escape_deterministic-v0_random',
     'gym_journey_escape_deterministic-v4_random',
     'gym_journey_escape_no_frameskip-v0_random',
     'gym_journey_escape_no_frameskip-v4_random',
     'gym_kangaroo-v0_random',
     'gym_kangaroo-v4_random',
     'gym_kangaroo_deterministic-v0_random',
     'gym_kangaroo_deterministic-v4_random',
     'gym_kangaroo_no_frameskip-v0_random',
     'gym_kangaroo_no_frameskip-v4_random',
     'gym_krull-v0_random',
     'gym_krull-v4_random',
     'gym_krull_deterministic-v0_random',
     'gym_krull_deterministic-v4_random',
     'gym_krull_no_frameskip-v0_random',
     'gym_krull_no_frameskip-v4_random',
     'gym_kung_fu_master-v0_random',
     'gym_kung_fu_master-v4_random',
     'gym_kung_fu_master_deterministic-v0_random',
     'gym_kung_fu_master_deterministic-v4_random',
     'gym_kung_fu_master_no_frameskip-v0_random',
     'gym_kung_fu_master_no_frameskip-v4_random',
     'gym_montezuma_revenge-v0_random',
     'gym_montezuma_revenge-v4_random',
     'gym_montezuma_revenge_deterministic-v0_random',
     'gym_montezuma_revenge_deterministic-v4_random',
     'gym_montezuma_revenge_no_frameskip-v0_random',
     'gym_montezuma_revenge_no_frameskip-v4_random',
     'gym_ms_pacman-v0_random',
     'gym_ms_pacman-v4_random',
     'gym_ms_pacman_deterministic-v0_random',
     'gym_ms_pacman_deterministic-v4_random',
     'gym_ms_pacman_no_frameskip-v0_random',
     'gym_ms_pacman_no_frameskip-v4_random',
     'gym_name_this_game-v0_random',
     'gym_name_this_game-v4_random',
     'gym_name_this_game_deterministic-v0_random',
     'gym_name_this_game_deterministic-v4_random',
     'gym_name_this_game_no_frameskip-v0_random',
     'gym_name_this_game_no_frameskip-v4_random',
     'gym_phoenix-v0_random',
     'gym_phoenix-v4_random',
     'gym_phoenix_deterministic-v0_random',
     'gym_phoenix_deterministic-v4_random',
     'gym_phoenix_no_frameskip-v0_random',
     'gym_phoenix_no_frameskip-v4_random',
     'gym_pitfall-v0_random',
     'gym_pitfall-v4_random',
     'gym_pitfall_deterministic-v0_random',
     'gym_pitfall_deterministic-v4_random',
     'gym_pitfall_no_frameskip-v0_random',
     'gym_pitfall_no_frameskip-v4_random',
     'gym_pong-v0_random',
     'gym_pong-v4_random',
     'gym_pong_deterministic-v0_random',
     'gym_pong_deterministic-v4_random',
     'gym_pong_no_frameskip-v0_random',
     'gym_pong_no_frameskip-v4_random',
     'gym_pooyan-v0_random',
     'gym_pooyan-v4_random',
     'gym_pooyan_deterministic-v0_random',
     'gym_pooyan_deterministic-v4_random',
     'gym_pooyan_no_frameskip-v0_random',
     'gym_pooyan_no_frameskip-v4_random',
     'gym_private_eye-v0_random',
     'gym_private_eye-v4_random',
     'gym_private_eye_deterministic-v0_random',
     'gym_private_eye_deterministic-v4_random',
     'gym_private_eye_no_frameskip-v0_random',
     'gym_private_eye_no_frameskip-v4_random',
     'gym_qbert-v0_random',
     'gym_qbert-v4_random',
     'gym_qbert_deterministic-v0_random',
     'gym_qbert_deterministic-v4_random',
     'gym_qbert_no_frameskip-v0_random',
     'gym_qbert_no_frameskip-v4_random',
     'gym_riverraid-v0_random',
     'gym_riverraid-v4_random',
     'gym_riverraid_deterministic-v0_random',
     'gym_riverraid_deterministic-v4_random',
     'gym_riverraid_no_frameskip-v0_random',
     'gym_riverraid_no_frameskip-v4_random',
     'gym_road_runner-v0_random',
     'gym_road_runner-v4_random',
     'gym_road_runner_deterministic-v0_random',
     'gym_road_runner_deterministic-v4_random',
     'gym_road_runner_no_frameskip-v0_random',
     'gym_road_runner_no_frameskip-v4_random',
     'gym_robotank-v0_random',
     'gym_robotank-v4_random',
     'gym_robotank_deterministic-v0_random',
     'gym_robotank_deterministic-v4_random',
     'gym_robotank_no_frameskip-v0_random',
     'gym_robotank_no_frameskip-v4_random',
     'gym_seaquest-v0_random',
     'gym_seaquest-v4_random',
     'gym_seaquest_deterministic-v0_random',
     'gym_seaquest_deterministic-v4_random',
     'gym_seaquest_no_frameskip-v0_random',
     'gym_seaquest_no_frameskip-v4_random',
     'gym_skiing-v0_random',
     'gym_skiing-v4_random',
     'gym_skiing_deterministic-v0_random',
     'gym_skiing_deterministic-v4_random',
     'gym_skiing_no_frameskip-v0_random',
     'gym_skiing_no_frameskip-v4_random',
     'gym_solaris-v0_random',
     'gym_solaris-v4_random',
     'gym_solaris_deterministic-v0_random',
     'gym_solaris_deterministic-v4_random',
     'gym_solaris_no_frameskip-v0_random',
     'gym_solaris_no_frameskip-v4_random',
     'gym_space_invaders-v0_random',
     'gym_space_invaders-v4_random',
     'gym_space_invaders_deterministic-v0_random',
     'gym_space_invaders_deterministic-v4_random',
     'gym_space_invaders_no_frameskip-v0_random',
     'gym_space_invaders_no_frameskip-v4_random',
     'gym_star_gunner-v0_random',
     'gym_star_gunner-v4_random',
     'gym_star_gunner_deterministic-v0_random',
     'gym_star_gunner_deterministic-v4_random',
     'gym_star_gunner_no_frameskip-v0_random',
     'gym_star_gunner_no_frameskip-v4_random',
     'gym_tennis-v0_random',
     'gym_tennis-v4_random',
     'gym_tennis_deterministic-v0_random',
     'gym_tennis_deterministic-v4_random',
     'gym_tennis_no_frameskip-v0_random',
     'gym_tennis_no_frameskip-v4_random',
     'gym_time_pilot-v0_random',
     'gym_time_pilot-v4_random',
     'gym_time_pilot_deterministic-v0_random',
     'gym_time_pilot_deterministic-v4_random',
     'gym_time_pilot_no_frameskip-v0_random',
     'gym_time_pilot_no_frameskip-v4_random',
     'gym_tutankham-v0_random',
     'gym_tutankham-v4_random',
     'gym_tutankham_deterministic-v0_random',
     'gym_tutankham_deterministic-v4_random',
     'gym_tutankham_no_frameskip-v0_random',
     'gym_tutankham_no_frameskip-v4_random',
     'gym_up_n_down-v0_random',
     'gym_up_n_down-v4_random',
     'gym_up_n_down_deterministic-v0_random',
     'gym_up_n_down_deterministic-v4_random',
     'gym_up_n_down_no_frameskip-v0_random',
     'gym_up_n_down_no_frameskip-v4_random',
     'gym_venture-v0_random',
     'gym_venture-v4_random',
     'gym_venture_deterministic-v0_random',
     'gym_venture_deterministic-v4_random',
     'gym_venture_no_frameskip-v0_random',
     'gym_venture_no_frameskip-v4_random',
     'gym_video_pinball-v0_random',
     'gym_video_pinball-v4_random',
     'gym_video_pinball_deterministic-v0_random',
     'gym_video_pinball_deterministic-v4_random',
     'gym_video_pinball_no_frameskip-v0_random',
     'gym_video_pinball_no_frameskip-v4_random',
     'gym_wizard_of_wor-v0_random',
     'gym_wizard_of_wor-v4_random',
     'gym_wizard_of_wor_deterministic-v0_random',
     'gym_wizard_of_wor_deterministic-v4_random',
     'gym_wizard_of_wor_no_frameskip-v0_random',
     'gym_wizard_of_wor_no_frameskip-v4_random',
     'gym_yars_revenge-v0_random',
     'gym_yars_revenge-v4_random',
     'gym_yars_revenge_deterministic-v0_random',
     'gym_yars_revenge_deterministic-v4_random',
     'gym_yars_revenge_no_frameskip-v0_random',
     'gym_yars_revenge_no_frameskip-v4_random',
     'gym_zaxxon-v0_random',
     'gym_zaxxon-v4_random',
     'gym_zaxxon_deterministic-v0_random',
     'gym_zaxxon_deterministic-v4_random',
     'gym_zaxxon_no_frameskip-v0_random',
     'gym_zaxxon_no_frameskip-v4_random',
     'image_celeba',
     'image_celeba32',
     'image_celeba64',
     'image_celeba_multi_resolution',
     'image_celebahq128',
     'image_celebahq128_dmol',
     'image_celebahq256',
     'image_celebahq256_dmol',
     'image_cifar10',
     'image_cifar100',
     'image_cifar100_plain',
     'image_cifar100_plain8',
     'image_cifar100_plain_gen',
     'image_cifar100_tune',
     'image_cifar10_plain',
     'image_cifar10_plain8',
     'image_cifar10_plain_gen',
     'image_cifar10_plain_gen_dmol',
     'image_cifar10_plain_gen_flat',
     'image_cifar10_plain_random_shift',
     'image_cifar10_tune',
     'image_cifar20',
     'image_cifar20_plain',
     'image_cifar20_plain8',
     'image_cifar20_plain_gen',
     'image_cifar20_tune',
     'image_fashion_mnist',
     'image_fsns',
     'image_imagenet',
     'image_imagenet224',
     'image_imagenet224_no_normalization',
     'image_imagenet256',
     'image_imagenet32',
     'image_imagenet32_gen',
     'image_imagenet32_small',
     'image_imagenet64',
     'image_imagenet64_gen',
     'image_imagenet64_gen_flat',
     'image_imagenet_multi_resolution_gen',
     'image_lsun_bedrooms',
     'image_mnist',
     'image_mnist_tune',
     'image_ms_coco_characters',
     'image_ms_coco_tokens32k',
     'image_text_ms_coco',
     'image_text_ms_coco_multi_resolution',
     'image_vqav2_rcnn_feature_tokens10k_labels3k',
     'image_vqav2_tokens10k_labels3k',
     'img2img_allen_brain',
     'img2img_allen_brain_dim16to16_paint1',
     'img2img_allen_brain_dim48to64',
     'img2img_allen_brain_dim8to32',
     'img2img_celeba',
     'img2img_celeba64',
     'img2img_cifar10',
     'img2img_cifar100',
     'img2img_imagenet',
     'lambada_lm',
     'lambada_lm_control',
     'lambada_rc',
     'lambada_rc_control',
     'languagemodel_de_en_fr_ro_wiki64k',
     'languagemodel_de_en_fr_ro_wiki64k_fitb_packed1k',
     'languagemodel_de_wiki32k',
     'languagemodel_de_wiki64k',
     'languagemodel_en_wiki32k',
     'languagemodel_en_wiki64k',
     'languagemodel_en_wiki64k_shorter',
     'languagemodel_en_wiki_lm_multi_nli_subwords',
     'languagemodel_en_wiki_lm_multi_nli_subwords64k',
     'languagemodel_en_wiki_lm_multi_nli_subwords_v2',
     'languagemodel_en_wiki_lm_short_multi_nli_subwords64k',
     'languagemodel_en_wiki_lm_squad_concat_subwords',
     'languagemodel_en_wiki_lm_summarize_cnndm_subwords',
     'languagemodel_en_wiki_lm_summarize_cnndm_subwords64k',
     'languagemodel_en_wiki_lm_summarize_frac10_cnndm_subwords64k',
     'languagemodel_en_wiki_lm_summarize_frac1_cnndm_subwords64k',
     'languagemodel_en_wiki_lm_summarize_frac20_cnndm_subwords64k',
     'languagemodel_en_wiki_lm_summarize_frac2_cnndm_subwords64k',
     'languagemodel_en_wiki_lm_summarize_frac50_cnndm_subwords64k',
     'languagemodel_en_wiki_lm_summarize_frac5_cnndm_subwords64k',
     'languagemodel_fr_wiki32k',
     'languagemodel_fr_wiki64k',
     'languagemodel_lm1b32k',
     'languagemodel_lm1b32k_packed',
     'languagemodel_lm1b8k',
     'languagemodel_lm1b8k_packed',
     'languagemodel_lm1b_characters',
     'languagemodel_lm1b_characters_packed',
     'languagemodel_lm1b_multi_nli',
     'languagemodel_lm1b_multi_nli_subwords',
     'languagemodel_lm1b_sentiment_imdb',
     'languagemodel_multi_wiki_translate',
     'languagemodel_multi_wiki_translate_fr',
     'languagemodel_multi_wiki_translate_packed1k',
     'languagemodel_multi_wiki_translate_packed1k_v2',
     'languagemodel_ptb10k',
     'languagemodel_ptb_characters',
     'languagemodel_ro_wiki32k',
     'languagemodel_ro_wiki64k',
     'languagemodel_wiki_noref_v128k_l1k',
     'languagemodel_wiki_noref_v32k_l16k',
     'languagemodel_wiki_noref_v32k_l1k',
     'languagemodel_wiki_noref_v8k_l16k',
     'languagemodel_wiki_noref_v8k_l1k',
     'languagemodel_wiki_scramble_l128',
     'languagemodel_wiki_scramble_l1k',
     'languagemodel_wiki_xml_v8k_l1k',
     'languagemodel_wiki_xml_v8k_l4k',
     'languagemodel_wikitext103',
     'languagemodel_wikitext103_characters',
     'languagemodel_wikitext103_l16k',
     'languagemodel_wikitext103_l4k',
     'librispeech',
     'librispeech_clean',
     'librispeech_clean_small',
     'librispeech_noisy',
     'librispeech_train_full_test_clean',
     'msr_paraphrase_corpus',
     'msr_paraphrase_corpus_characters',
     'multi_nli',
     'multi_nli_characters',
     'multi_nli_shared_vocab',
     'multi_nli_text2text',
     'multi_nli_text2text_multi64k_packed1k',
     'multi_nli_wiki_lm_multi_vocab64k',
     'multi_nli_wiki_lm_shared_vocab',
     'multi_nli_wiki_lm_shared_vocab64k',
     'ocr_test',
     'paraphrase_generation_ms_coco_problem1d',
     'paraphrase_generation_ms_coco_problem1d_characters',
     'paraphrase_generation_ms_coco_problem2d',
     'paraphrase_generation_ms_coco_problem2d_characters',
     'parsing_english_ptb16k',
     'parsing_english_ptb8k',
     'parsing_icelandic16k',
     'program_search_algolisp',
     'programming_desc2code_cpp',
     'programming_desc2code_py',
     'question_nli',
     'question_nli_characters',
     'quora_question_pairs',
     'quora_question_pairs_characters',
     'reverse_sequence',
     'reverse_sequence_small',
     'rte',
     'rte_characters',
     'sci_tail',
     'sci_tail_characters',
     'sci_tail_shared_vocab',
     'sentiment_imdb',
     'sentiment_imdb_characters',
     'sentiment_sst_binary',
     'sentiment_sst_binary_characters',
     'sentiment_yelp_full',
     'sentiment_yelp_full_characters',
     'sentiment_yelp_polarity',
     'sentiment_yelp_polarity_characters',
     'squad',
     'squad_concat',
     'squad_concat_multi64k',
     'squad_concat_positioned',
     'squad_concat_shared_vocab',
     'squad_text2text',
     'squad_text2text_multi64k_packed1k',
     'stanford_nli',
     'stanford_nli_characters',
     'stanford_nli_shared_vocab',
     'stanford_nli_wiki_lm_shared_vocab',
     'stanford_nli_wiki_lm_shared_vocab64k',
     'style_transfer_modern_to_shakespeare',
     'style_transfer_modern_to_shakespeare_characters',
     'style_transfer_shakespeare_to_modern',
     'style_transfer_shakespeare_to_modern_characters',
     'summarize_cnn_dailymail32k',
     'summarize_cnn_dailymail_multi64k_packed1k',
     'summarize_cnn_dailymail_wiki_lm_multi_vocab64k',
     'summarize_cnn_dailymail_wiki_lm_shared_vocab',
     'summarize_cnn_dailymail_wiki_lm_shared_vocab64k',
     'summarize_frac0p1_cnn_dailymail_wiki_lm_shared_vocab64k',
     'summarize_frac10_cnn_dailymail_wiki_lm_shared_vocab64k',
     'summarize_frac1_cnn_dailymail_wiki_lm_shared_vocab64k',
     'summarize_frac20_cnn_dailymail_wiki_lm_shared_vocab64k',
     'summarize_frac2_cnn_dailymail_wiki_lm_shared_vocab64k',
     'summarize_frac50_cnn_dailymail_wiki_lm_shared_vocab64k',
     'summarize_frac5_cnn_dailymail_wiki_lm_shared_vocab64k',
     'summarize_frac_cnn_dailymail_wiki_lm_shared_vocab64k',
     'sva_language_modeling',
     'sva_number_prediction',
     'text2text_copyable_tokens',
     'text2text_tmpdir',
     'text2text_tmpdir_tokens',
     'timeseries_synthetic_data_series10_samples100k',
     'timeseries_toy_problem',
     'timeseries_toy_problem_no_inputs',
     'tiny_algo',
     'translate_encs_wmt32k',
     'translate_encs_wmt_characters',
     'translate_ende_pc32k',
     'translate_ende_pc_clean32k',
     'translate_ende_wmt32k',
     'translate_ende_wmt32k_packed',
     'translate_ende_wmt8k',
     'translate_ende_wmt8k_packed',
     'translate_ende_wmt_characters',
     'translate_ende_wmt_clean32k',
     'translate_ende_wmt_clean_pc32k',
     'translate_ende_wmt_clean_pc_clean32k',
     'translate_ende_wmt_multi64k',
     'translate_ende_wmt_multi64k_packed1k',
     'translate_ende_wmt_pc32k',
     'translate_ende_wmt_pc_clean32k',
     'translate_enet_wmt32k',
     'translate_enet_wmt_characters',
     'translate_enfr_wmt32k',
     'translate_enfr_wmt32k_packed',
     'translate_enfr_wmt32k_with_backtranslate_en',
     'translate_enfr_wmt32k_with_backtranslate_fr',
     'translate_enfr_wmt8k',
     'translate_enfr_wmt_characters',
     'translate_enfr_wmt_multi64k',
     'translate_enfr_wmt_multi64k_packed1k',
     'translate_enfr_wmt_small32k',
     'translate_enfr_wmt_small8k',
     'translate_enfr_wmt_small_characters',
     'translate_enid_iwslt32k',
     'translate_enmk_setimes32k',
     'translate_enmk_setimes_characters',
     'translate_enro_wmt32k',
     'translate_enro_wmt8k',
     'translate_enro_wmt_characters',
     'translate_enro_wmt_multi64k',
     'translate_enro_wmt_multi_small64k',
     'translate_enro_wmt_multi_tiny64k',
     'translate_enro_wmt_multi_tiny64k_packed1k',
     'translate_envi_iwslt32k',
     'translate_enzh_wmt32k',
     'translate_enzh_wmt8k',
     'video_bair_robot_pushing',
     'video_bair_robot_pushing_with_actions',
     'video_google_robot_pushing',
     'video_stochastic_shapes10k',
     'wiki_revision',
     'wiki_revision_packed1k',
     'wiki_revision_packed256',
     'wikisum_commoncrawl',
     'wikisum_commoncrawl_lead_section',
     'wikisum_web',
     'wikisum_web_lead_section',
     'winograd_nli',
     'winograd_nli_characters',
     'wsj_parsing']




```python
# Fetch the MNIST problem
mnist_problem = problems.problem("image_mnist")
# The generate_data method of a problem will download data and process it into
# a standard format ready for training and evaluation.
mnist_problem.generate_data(data_dir, tmp_dir)
```

    W0822 15:58:08.634740  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\data_generators\generator_utils.py:226: The name tf.gfile.MakeDirs is deprecated. Please use tf.io.gfile.makedirs instead.
    
    W0822 15:58:08.637740  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\data_generators\generator_utils.py:228: The name tf.gfile.Exists is deprecated. Please use tf.io.gfile.exists instead.
    
    W0822 15:58:09.688677  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\data_generators\generator_utils.py:164: The name tf.python_io.TFRecordWriter is deprecated. Please use tf.io.TFRecordWriter instead.
    
    W0822 15:58:39.080926  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\data_generators\generator_utils.py:183: The name tf.gfile.Rename is deprecated. Please use tf.io.gfile.rename instead.
    
    W0822 15:58:43.596652  9748 deprecation.py:323] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\data_generators\generator_utils.py:469: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use eager execution and: 
    `tf.data.TFRecordDataset(path)`
    W0822 15:58:44.184562  9748 deprecation_wrapper.py:119] From C:\Users\admin\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\data_generators\generator_utils.py:513: The name tf.gfile.Remove is deprecated. Please use tf.io.gfile.remove instead.
    
    


```python
# Now let's see the training MNIST data as Tensors.
mnist_example = tfe.Iterator(mnist_problem.dataset(Modes.TRAIN, data_dir)).next()
image = mnist_example["inputs"]
label = mnist_example["targets"]

plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap('gray'))
print("Label: %d" % label.numpy())
```

    Label: 2
    


![png](output_9_1.png)



```python
# Fetch the problem
ende_problem = problems.problem("translate_ende_wmt32k")

# Copy the vocab file locally so we can encode inputs and decode model outputs
# All vocabs are stored on GCS
vocab_name = "vocab.translate_ende_wmt32k.32768.subwords"
vocab_file = os.path.join(gs_data_dir, vocab_name)
!gsutil cp {vocab_file} {data_dir}

# Get the encoders from the problem
encoders = ende_problem.feature_encoders(data_dir)

# Setup helper functions for encoding and decoding
def encode(input_str, output_str=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"inputs": batch_inputs}

def decode(integers):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if 1 in integers:
    integers = integers[:integers.index(1)]
  return encoders["inputs"].decode(np.squeeze(integers))
```

    'gsutil' 不是内部或外部命令，也不是可运行的程序
    或批处理文件。
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-7-60ff6a5887f7> in <module>
          9 
         10 # Get the encoders from the problem
    ---> 11 encoders = ende_problem.feature_encoders(data_dir)
         12 
         13 # Setup helper functions for encoding and decoding
    

    ~\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\data_generators\text_problems.py in feature_encoders(self, data_dir)
        197 
        198   def feature_encoders(self, data_dir):
    --> 199     encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
        200     encoders = {"targets": encoder}
        201     if self.has_inputs:
    

    ~\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\data_generators\text_problems.py in get_or_create_vocab(self, data_dir, tmp_dir, force_get)
        242       if force_get:
        243         vocab_filepath = os.path.join(data_dir, self.vocab_filename)
    --> 244         encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
        245       else:
        246         other_problem = self.use_vocab_from_other_problem
    

    ~\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\data_generators\text_encoder.py in __init__(self, filename)
        489     self.filename = filename
        490     if filename is not None:
    --> 491       self._load_from_file(filename)
        492     super(SubwordTextEncoder, self).__init__()
        493 
    

    ~\.conda\envs\lstm-crf\lib\site-packages\tensor2tensor\data_generators\text_encoder.py in _load_from_file(self, filename)
        937     """Load from a vocab file."""
        938     if not tf.gfile.Exists(filename):
    --> 939       raise ValueError("File %s not found" % filename)
        940     with tf.gfile.Open(filename) as f:
        941       self._load_from_file_object(f)
    

    ValueError: File C:\Users\admin/t2t/data\vocab.translate_ende_wmt32k.32768.subwords not found



```python

```
