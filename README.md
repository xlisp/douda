# 测试的标准: anaconda3可以跑下面这段代码, 先安装anaconda3,再安装cuda

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

# zshrc

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


alias proxy_clj=' ssh -p 26401 -N -R 8888:localhost:9999 clojure@67.216.200.53 '

alias http='  python -m http.server 2222 '
alias e=' emacs  -q -l ~/clojure_emacs/init.el '
alias gpu_t=' nvidia-smi -l '

##export CUDA_VISIBLE_DEVICES=0

alias gch=' git checkout '


```
