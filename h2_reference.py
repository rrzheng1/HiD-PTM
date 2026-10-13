import torch
from torch import nn
from torch.nn import functional as F
from baseline import SiteDiseaseModel as Original

class SiteDiseaseModel(Original):
    def __init__(self,text,edges,blosum,variant='H1'):
        super().__init__(text,edges,blosum);self.variant=variant
        self.multiscale=variant in ['H1','H3'];self.hierarchy=variant in ['H2','H3']
        if self.multiscale:
            self.scale_router=nn.Linear(128,3)
            # Equal initial weighting reproduces the original multiscale fusion.
            nn.init.zeros_(self.scale_router.weight);nn.init.zeros_(self.scale_router.bias)
        if self.hierarchy:
            self.protein_disease_gate=nn.Linear(384,128)
            self.site_protein_gate=nn.Linear(384,128)
            self.hierarchy_norm=nn.LayerNorm(128)

    def forward(self,esm,global_esm,aa,mask,ptm,disease,disease_table=None,return_aux=False):
        d=(self.disease_table() if disease_table is None else disease_table)[disease]
        local=torch.cat([F.one_hot(aa,21).float(),self.blosum[aa]],dim=-1)*mask.unsqueeze(-1)
        branches=[F.gelu(conv(local.transpose(1,2))).transpose(1,2) for conv in self.local]
        weights=None
        if self.multiscale:
            # Decompose the SAME original projection into three scale blocks.
            projected=torch.stack([F.linear(branch,self.local_projection.weight[:,i*32:(i+1)*32]) for i,branch in enumerate(branches)],dim=1)
            weights=torch.softmax(self.scale_router(d),dim=-1)
            local_fused=(projected*(3*weights[:,:,None,None])).sum(1)+self.local_projection.bias
        else:local_fused=self.local_projection(torch.cat(branches,dim=-1))
        protein=self.global_projection(global_esm.float())
        tokens=self.site_norm(self.esm_projection(esm.float())+local_fused+protein.unsqueeze(1)+
                              self.ptm_projection(self.ptm(ptm)).unsqueeze(1)+self.position)
        context,attention=self.attention(d.unsqueeze(1),tokens,tokens,key_padding_mask=~mask,need_weights=return_aux,average_attn_weights=False)
        site=self.condition_norm(tokens[:,15]+context[:,0]);protein_gate=None;site_gate=None
        if self.hierarchy:
            protein_gate=torch.sigmoid(self.protein_disease_gate(torch.cat([protein,d,protein*d],dim=-1)))
            disease_protein=protein_gate*protein
            site_gate=torch.sigmoid(self.site_protein_gate(torch.cat([site,disease_protein,d],dim=-1)))
            site=self.hierarchy_norm(site+site_gate*disease_protein)
        gate=torch.sigmoid(self.gate(torch.cat([site,d,site*d],dim=-1)))
        logits=self.head(torch.cat([site*d,gate*site,(1-gate)*d],dim=-1)).squeeze(-1)
        if return_aux:return logits,dict(scale_weights=weights,protein_gate=protein_gate,site_gate=site_gate,residue_attention=attention)
        return logits
