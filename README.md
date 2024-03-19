English Version
# pytorch离散数据特征抽取&模型训练框架
pytorch的优点是非常灵活，上手方便，但是在日常的使用中会发现在训练过程中经常需要写重复代码，并且代码量有非常大的问题。如果使用pytorch-lighting等训练框架会发现抽象度因为过于高，导致出现了异常非常难debug，排查问题的过程非常复杂。

同时在面对一些小场景例如点击率预估等，我们拿到的特征通常是一些数值型或者枚举型特征。这种数据通常需要自己定义DataSet进行处理，这样本来就不轻松的代码量更加雪上加霜。

针对以上问题，本框架定制了一个通用的离散数据特征处理和训练框架，特征处理部分大家无需做过多的开发，只需要进行简单的配置即可，方便大家可以快速处理**数值型和枚举型特征**，并且可以快速上手写出pytorch模型，同时可以定制化自己的模型，评价指标。

*目前本框架尚未支持文本类型数据，如有需要请联系我*
## 使用方法
### 运行说明
本框架支持模型训练和模型预测
#### 模型训练
模型训练需要运行train.py，参数说明可以通过进行查看，具体的运行参数说明可以参考[运行参数说明](#运行参数说明)
```
python train.py --h
```
不论运行什么模型，都需要指定**训练数据，特征配置，模型名称，label名称，训练mode**
```
python train.py --fields=conf/test.json --train_file=pandas://data/new_train.csv --header_num=0 --label_key=is_cost_raise --model_name=DNN --mode=train
```
其中读取训练数据需要指定文件类型，**pandas文件需要增加前缀pandas://，numpy文件需要增加前缀file://，特征配置文件必须是json的格式**
如果你不想从头训练，想要加载一个之前训练过的模型，可以指定load_model_path
```
python train.py --fields=conf/test.json --train_file=pandas://data/new_train.csv --header_num=0 --label_key=is_cost_raise --model_name=DNN --mode=train --load_model_path=model_path
```
#### 模型预测
模型预测需要指定**预测数据，特征配置，模型名称，label名称，训练mode，模型load地址**
```
python infer.py --fields=conf/test.json --valid_file=pandas://data/new_train.csv --header_num=0 --label_key=is_cost_raise --model_name=DNN --mode=infer --load_model_path=model_path
```
### 运行参数说明
在训练或预测过程中我们需要指定运行参数来调整模型的训练或预测过程。
#### 基本参数说明
不论是训练还是预测都有一些必要的参数：
- mode: 训练模式，可以选择 train, infer
- fields: 模型特征配置，必须是json的格式
- model_name: 要加载的模型名称
- label_key: 必要参数，模型的目标字段
##### 训练参数
- train_file: 必要参数，训练文件，支持numpy和pandas格式，numpy文件以"file://"开头，pandas文件以"pandas://"开头
- header_num: 如果训练文件是pandas，需要指定训练的文件的header是第几行
- do_eval: 是否验证模型
- valid_file: 如果do_eval是True的话，那么需要配置valid_file，配置方式和train_file一致
- init_score_boarder:是否需要初始化score boarder评价指标
- boarder_name:如果init_score_boarder是True，需要定义评价指标名称
- do_summary: 是否需要summary
- summary_path: tensor board地址，如果do_summary为True需要配置
- total_epoch:必要参数，整体训练的轮数
- batch_size: 必要参数，训练时一次计算的数据量
- learning_rate:必要参数，学习率
- decay_rate: 必要参数，权值正则化参数
- decay_epoch: 必要参数，正则化的轮数
- model_path: 模型存储的位置
- output_path: 验证结果输出的位置
##### 预测参数
- valid_file: 必要参数，预估文件，支持numpy和pandas格式，numpy文件以"file://"开头，pandas文件以"pandas://"开头
- batch_size: 必要参数，训练时一次计算的数据量
- output_path: 预估结果输出的位置
- load_model_path: 要加载的模型位置

### 模型特征配置
模型特征配置可参考conf目录下的配置文件。特征组织按照json的格式进行，每一个字段配置为一个json类型。
必须配置的字段为：**key,type,dtype**
下面对特征配置字段进行说明
- key: 表示这个字段的名称，是数据的唯一id标志，不能有重复
- type: 数据类型有 id, sparse, dense, sparse_seq, target, label和pass九种枚举类型。下面对这几种类型的配置进行说明
    - id: id类，特征必须是数字类型，在使用中不会被用到，只是一个记录的意义
    - sparse: 枚举类特征，如果是string类型需要配制hash type，可配置字段如下
        - hash_type: 必要字段，哈希类型有sklearn,tensorflow, remainder三种选择，一般推荐用sklearn
        - embed_key: 必要字段，embed dict 的key, 一般和key同名即可
        - i_dim: embed dim表示embedding dict有多少个id
        - embedding dim需要在args.emb_dim中配置，默认是8
        - dtype: 数据类型，必须为int64
    - sparse_seq: 多个枚举类型列表类特征，例如1,2,3,4，这个默认会取maxpooling，如果有需要可以自己在model文件中自行修改处理模式
        - hash_type: 表示哈希的类型，推荐为string
        - embed_key: 必要字段，embed dict 的key, 一般和key同名即可
        - i_dim: embed dim表示embedding dict有多少个id
        - embedding dim需要在args.emb_dim中配置，默认是8
        - dtype: 数据类型，必须为int64
        - padding_first: 如果当前序列的长度小于seq_length，则对前面的数字进行padding
        - remain_last: 如果当前序列长度大于seq_length，remain_last为true表示保留后seq_length的数字，否则保留前seq_length的数字
        - seq_length: 序列长度，如果当前序列的长度小于seq_length，则对前面的数字进行padding
    - dense: 数值类型特征，表示具体的数字，例如3.2，1.8等
        - dtype: 数据类型，可以为int64也可以为float32
        - trans: 数据处理函数，可以支持自定义函数，目前集成了一些处理函数
            - ln: 取对数
            - log2: 取2的对数
            - log10: 取10的对数
            - div: 除某一个数字，具体用法为div2,div100等
            - clip: 用法为clip,m,n表示对数字[m,n]的映射，小于等于m的统一为m，大于等于n的统一为n
            - 自定义: 如果需要自定义数据处理类型，需要在funcs中定义好处理函数，在trans中指定函数位置和函数名称。例如函数test位于funcs.dense_helper中，那么想要对特征进行test处理，那么可以对trans设置为"dense_helper.test"
        - length: 如果特征是一个数值类型的列表，可以在此指定列表的长度
            - sep: 特征分隔符，本框架会以此分隔符对列表进行分隔
            - range: 需要取得数字范围，格式为m,n表示取[m,n]范围内的数字
    - target: 模型label，对于回归任务需要配置为target
    - label: 模型label，对于分类任务需要配置为label
    - pass: 略过这个字段，对于不想要的字段可以配置pass
### 模型开发
如果要实现新的模型，首先需要在models目录下新增模型文件，然后需要在models/__init__.py中进行模型注册。
#### 新增模型
首先我们需要新增一个python文件，定义需要新增的模型。
我们在models/BaseModel.py中**对模型的特征配置文件按照dense和sparse特征进行了解析，数值类型特征保存在dense_keys，维度为dense_dim，枚举类型保存在linear_keys中，维度为embed_dim**，建议新增模型继承BaseModel，可基于以上信息进行模型开发。
下面是一个建议的模型开发代码：
```python
from ..BaseModel import BaseModel

class Model(BaseModel):
     def __init__(self, args):
        super(Model, self).__init__(args)
        # 计算模型的输入维度
        # linear_keys 为枚举类型特征
        # dense_keys 为数值类型特征
        # merge_dim 为sparse特征加上dense特征之后的维度
        merge_dim = 0

        if len(self.linear_keys) > 0:
            self.linear_module = nn.Linear(self.embed_dim, args.hidden_dim)
            merge_dim += args.hidden_dim
        if len(self.dense_keys) > 0:
            self.dense_module = nn.Linear(self.dense_dim, args.hidden_dim)
            merge_dim += args.hidden_dim
        # 模型的核心部分，开始构建模型
    
    def forward(self, inputs):
        # e_list为枚举类型特征，对于序列型特征当前是采用max pool，如果有其他需求可以自行修改
        # d_list为数值类型特征
        # x_list会对e_list和d_list进行拼接，作为输入x组成模型输入
        e_list = list()
        d_list = list()
        for k in self.linear_keys:
            ek = self.embed_keys[k]
            x = inputs[k]
            e = F.dropout(self.embed_dict[ek](x), self.dropout, self.training)
            if len(e.shape) == 3 and e.shape[1] == 1:
                e = e.squeeze(1)
            elif len(e.shape) == 4:
                e = F.max_pool2d(e, (e.shape[-2], 1))
                e = e.squeeze(1).squeeze(1)
            else:
                e = torch.mean(e.squeeze(1), dim=1)
            e_list.append(e)
        for k in self.dense_keys:
            x = inputs[k]
            d_list.append(x)
        x_list = list()
        if len(e_list) > 0:
            e = torch.concat(e_list, dim=-1)
            x_list.append(self.linear_module(e))
        if len(d_list) > 0:
            d = torch.concat(d_list, dim=-1)
            x_list.append(self.dense_module(d))
        x = torch.concat(x_list, dim=-1)
        # 按照模型的结构开始处理输入
```
#### 模型注册
开发完模型之后需要在models/__init__.py中进行模型注册，具体注册方式也可以参考已经实现的模型。
### 新增评价指标
如果在训练和预测过程中需要衡量除loss之外的模型指标，可以在模型训练中设置init_score_boarder参数，同时指定score boarder名称。
如果需要新增评价指标，可以在models/score_boarder目录下新增一个文件，然后在models/__init__.py进行指标注册。
#### 新增指标
新增指标需要在models/score_boarder目录下新增一个文件，为方便大家开发，我们实现了一个BaseScoreBoarder.py，后续开发推荐继承该父类。当前推荐的score_boarder代码框架如下：
```python
from .BaseScoreBoarder import BaseScoreBoarder

class BinaryScoreBoarder(BaseScoreBoarder):
    # labels是数据的label
    # outputs是模型的输出
    def __init__(self):
        self.labels = list()
        self.outputs = list()

    def clear(self):
        # 对于多轮训练，需要重新清理labels和outputs，已经在BaseScoreBoarder中实现，如果不继承需要自己实现
        self.labels = list()
        self.outputs = list()

    def call_metric(self):
        # 计算新的评价指标
        return

    def log(self, prefix=None):
        # 如果需要打印评价指标可以在此实现
        return
```
#### 指标注册
开发完模型之后需要在models/__init__.py中进行指标注册，具体注册方式也可以参考已经实现的指标。
### 一个小小的使用样例
为了方便大家能够更快的开始，我们在data里准备了两份点击率预估的样例文件，[文件来源](https://github.com/reczoo/FuxiCTR/tree/main/data/tiny_csv)，特征抽取配置在conf/train_sample.json中，训练可以执行命令：
```
python3 train.py --fields=file://conf/test.json \
    --train_file=pandas://data/train_sample.csv \
    --valid_file=pandas://data/test_sample.csv \
    --header_num=0 \
    --label_key=clk \
    --emb_dim=16 \
    --do_eval=true \
    --model_name=DNN \
    --model_path=./model_result \
    --learning_rate=0.0001 \
    --total_epoch=30 \
    --decay_epoch=5 \
    --decay_rate=1e-4 \
    --init_score_boarder=true \
    --boarder_name=BinaryScoreBoarder \
    --file_delimiter=,
```
当然在运行过程中可能会因为数据过小，导致模型运行出错，此时建议收集更多的数据，增加数据规模，来提高训练效果。
### 未来计划
1. 添加文本类型数据支持
2. 添加lr_scheduler的可定制化支持
3. 多卡并行方法支持
