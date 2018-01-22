# Hylang Lisp 的语音识别: 先py跑起来,用hylangλ化,就像Clojureλ化Spark和安卓,前端一样,任何领域的Lispλ化

* douda.py跑在CPU上正常,但是在GPU上跑,是报错的: douda_gpu_erro.txt
* 只要能描述清楚了(文学编程的最高境界)，就能lispλ化=>只要特征能描述清楚了,就能hylisp可微分化
* Clojure和Java的互操作,迁移到Hylisp和Python的互操作: `test_hy.hy`编译生成pyc`test_hy.pyc`

### 测试的标准: anaconda3可以跑下面这段代码, 先安装anaconda3,再安装cuda(其实就是先有anaconda3的PATH变量,CUDA能找到就行了)

```python

import tensorflow as tf

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
print(sess.run(c))

```

### zshrc: 如果CUDA找不到PATH里面有anaconda3,就会去找系统自带的python，所以只有系统自带的python可以用GPU跑，而anaconda3不能(只能跑在CPU上)

```bash

export PATH=/home/hylisp/anaconda3/bin:$PATH
### anaconda3 死活都用不了GPU ==>> Ubuntu自带的Python可以


export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda

alias vv=' vi ~/.zshrc ; source ~/.zshrc '

## pip install --upgrade tensorflow-gpu

alias gd='git diff '
alias gs='git status '


## https://github.com/globus/globus-jupyter-notebooks
alias pyweb=' jupyter notebook --ip="*" --no-browser '
## .e.g: ➜  learn git:(master) ✗ jupyter notebook --ip="*" --no-browser



alias http='  python -m http.server 2222 '
alias e=' emacs  -q -l ~/clojure_emacs/init.el '
alias gpu_t=' nvidia-smi -l '

##export CUDA_VISIBLE_DEVICES=0

alias gch=' git checkout '


```
