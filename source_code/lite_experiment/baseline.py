import torch
from torch import nn
from torch.nn import functional as F

class SiteDiseaseModel(nn.Module):
    def __init__(self,text,edges,blosum,interaction='cross',ontology=True):
        super().__init__(); self.interaction=interaction;self.ontology=ontology
        self.register_buffer('text',text.float());self.register_buffer('blosum',blosum.float())
        n=len(text);degree=torch.bincount(edges[0],minlength=n).float()
        value=(degree[edges[0]]*degree[edges[1]]).clamp_min(1).rsqrt()
        self.register_buffer('adjacency',torch.sparse_coo_tensor(edges,value,(n,n),check_invariants=True).coalesce())
        self.text_projection=nn.Linear(text.shape[1],128)
        self.gcn=nn.ModuleList([nn.Linear(128,128),nn.Linear(128,128)])
        self.graph_norm=nn.ModuleList([nn.LayerNorm(128),nn.LayerNorm(128)])
        self.esm_projection=nn.Linear(1280,128);self.global_projection=nn.Linear(1280,128)
        self.local=nn.ModuleList([nn.Conv1d(41,32,k,padding=k//2) for k in [3,5,7]])
        self.local_projection=nn.Linear(96,128)
        self.ptm=nn.Embedding(5,32);self.ptm_projection=nn.Linear(32,128)
        self.position=nn.Parameter(torch.empty(31,128));nn.init.normal_(self.position,std=.02)
        self.site_norm=nn.LayerNorm(128)
        self.attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
        self.condition_norm=nn.LayerNorm(128)
        self.gate=nn.Linear(384,128)
        self.head=nn.Sequential(nn.Linear(384 if interaction=='cross' else 256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))

    def disease_table(self):
        d=F.gelu(self.text_projection(self.text))
        if self.ontology:
            for layer,norm in zip(self.gcn,self.graph_norm):
                d=norm(d+F.dropout(F.gelu(layer(torch.sparse.mm(self.adjacency,d))),p=.1,training=self.training))
        return d

    def forward(self,esm,global_esm,aa,mask,ptm,disease,disease_table=None):
        d=(self.disease_table() if disease_table is None else disease_table)[disease]
        onehot=F.one_hot(aa,21).float();local=torch.cat([onehot,self.blosum[aa]],dim=-1)
        local=local*mask.unsqueeze(-1)
        cnn=torch.cat([F.gelu(conv(local.transpose(1,2))) for conv in self.local],dim=1).transpose(1,2)
        tokens=self.site_norm(self.esm_projection(esm.float())+self.local_projection(cnn)+
                              self.global_projection(global_esm.float()).unsqueeze(1)+
                              self.ptm_projection(self.ptm(ptm)).unsqueeze(1)+self.position)
        if self.interaction=='cross':
            context,_=self.attention(d.unsqueeze(1),tokens,tokens,key_padding_mask=~mask,need_weights=False)
            s=self.condition_norm(tokens[:,15]+context[:,0])
            gate=torch.sigmoid(self.gate(torch.cat([s,d,s*d],dim=-1)))
            pair=torch.cat([s*d,gate*s,(1-gate)*d],dim=-1)
        else:
            pooled=(tokens*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            pair=torch.cat([self.condition_norm(tokens[:,15]+pooled),d],dim=-1)
        return self.head(pair).squeeze(-1)
