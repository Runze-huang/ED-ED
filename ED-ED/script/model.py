import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
"""
为何编码器的gru中不用输入h0   答：nn.GRU是完整的GUR，直接输入一个序列数据即可，而nn.GRUCell是一个单元，要输入每个step的数据，和上一个状态的h(n-1)
batch为1时、为10时grucell分别怎么处理   答：batch为10时，可能先输入第一段旋律产生 h，再输入第二段，知道十段旋律全部输入产生10个 h
"""

class VAE(nn.Module):
    def __init__(self, #  (36,2048,51,12,100)
                 roll_dims,  #钢琴卷维数
                 hidden_dims, #隐藏层维数
                 infor_dims, #输入信息维数
                 n_step,  #一段旋律的step数
                 condition_dims, #和弦维数
                 k=700):
        super(VAE, self).__init__()
        #在解码器中生成旋律
        self.grucell_1 = nn.GRUCell(roll_dims + infor_dims, hidden_dims) #输入维度和隐藏层维度
        self.grucell_2 = nn.GRUCell(hidden_dims, hidden_dims)
        #初始化grucell_1的h0
        self.linear_init_1 = nn.Linear(infor_dims, hidden_dims)
        #生成一个step的旋律
        self.linear_out_1 = nn.Linear(hidden_dims, roll_dims)

        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 1 #调整使用teacher_force的概率
        self.sample = None
        self.iteration = 0
        self.k = torch.FloatTensor([k])
    #把x变成ont_hot向量
    def _sampling(self, x): 
        idx = x.max(1)[1] #返回最大的值的位置索引
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    #重建旋律
    def final_decoder(self, infor,length):  #输入z、节奏、和弦信息
        silence = torch.zeros(self.roll_dims)
        silence[-1] = -1
        #初始化第一个音符
        out = torch.zeros((infor.size(0), self.roll_dims))  #（batch , 36）
        out[:, -1] = 1.
        x, hx = [], [None, None]
        t = torch.tanh(self.linear_init_1(infor))  #(batch，z1_dims)--> (batch, hidden_dims)，再经过tanh变化
        hx[0] = t  # (batch, hidden_dims)   
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):  #这里逐次生成100 step的音乐
            out = torch.cat([out, infor], 1) # out shape为 （batch ，36+51）
            hx[0] = self.grucell_1(out, hx[0])  #把生成的上一个(音符+节奏+z+和弦)，和自身的上一个状态h(n-1)当作输入，生成下一个状态h(n)
            #第一个H(0)复制h(0)
            if i == 0: 
                hx[1] = hx[0]
            hx[1] = self.grucell_2(hx[0], hx[1]) #把grucell_1生成的状态h(n)，和自身的上一个状态H(n-1)当作输入，生成下一个状态H(n)
            out = F.log_softmax(self.linear_out_1(hx[1]), 1) #把grucell_2生成的状态H(n),经过softmax、ln处理后当作下一个音符
            x.append(out)
            #有一定概率把真实值当作输入，在刚开始加快训练
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:  #直接取真实值
                    out = self.sample[:, i, :]
                else:  #不取真实值，把生成的结果变成one_hot
                    out = self._sampling(out)
                self.eps = self.k / (self.k + torch.exp(self.iteration / self.k)) #从5000开始下降很快，iteration=8000时eps为0.25，iteration=12000时eps为0.006
            else:
                out = self._sampling(out)
        x = torch.stack(x, 1)

        #遍历每个step，把多余的置零
        
        for j in range(x.shape[0]):
            #如果长度小于100则把小于的部分变成休止符
            if length[j] != self.n_step :
                x[j,length[j]-self.n_step:] = silence
        
        return x

    #给出z1，z2，chroma，用来重建旋律
    def decoder(self, pitch, rhythm, condition=None):
        infor = torch.cat((pitch , rhythm)).unsqueeze(0)
        length = torch.sum(rhythm).int().unsqueeze(0)
        return self.final_decoder(infor, length)

    def forward(self, infor,x,length):
        #infor = infor / 10
        #print(infor[1])
        if self.training:
            #得到旋律样本和节奏样本
            self.sample = x
            self.iteration += 1
        recon = self.final_decoder(infor,length) #得到重建的旋律
        return recon