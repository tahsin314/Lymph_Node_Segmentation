import torch
import torch.nn as nn
import torch.utils.data as data
import math
from pytorch_model_summary import summary
from torchviz import make_dot

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransfomerModel(nn.Module):
    def __init__(self, num_channels, seq_len, downsample_factor=1, dim=192, depth=12, 
                 head_size=32, k_size=5, stride = 5, num_classes=3,  **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.downsample_factor = downsample_factor
        self.stride = stride
        self.emb = nn.Conv1d(num_channels, dim, k_size, stride) # 
        self.pos_enc = SinusoidalPosEmb(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, 
                                       nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.2, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
        self.conv2 = nn.Conv1d(dim, 1, k_size, stride)
        self.proj_out = nn.Linear(seq_len//(stride*stride*downsample_factor), num_classes)
    
    def forward(self, x):
        pos = torch.arange(self.seq_len//(self.stride*self.downsample_factor), device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos).transpose(2, 1)
        x = self.emb(x)
        x = x + pos
        x = self.transformer(x.transpose(2, 1))
        x = self.conv2(x.transpose(2, 1))
        x = x.view(x.size(0), -1)
        x = self.proj_out(x)
        
        return x

if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    model = TransfomerModel(num_channels=3, seq_len=900, downsample_factor=3,
                            dim=48, head_size=6, num_classes=6)
    data = torch.randn(4, 3, 900)
    out = model(data)
    print(out.shape)
    print(summary(model, data, show_input=False))
    dot = make_dot(model(data), params=dict(model.named_parameters()))
    dot.format = 'png'  # You can change the format as needed
    dot.render('../model_graphs/mini_transformer_graph')