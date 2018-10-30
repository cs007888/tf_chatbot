Chatbot Model

参照https://github.com/tensorflow/nmt

重新训练:
(1) 只需要修改data/origin下的数据集文件和hparams
(2) 删除data/generated下的文件(会重新自动生成)
(3) 运行train.py