import torch
from torch.nn import functional as F
from h2_reference import SiteDiseaseModel as H2

class SiteDiseaseModel(H2):
    def __init__(self,text,edges,blosum,variant='F0'):
        # Initialize the exact same H2 parameters in all configurations.
        super().__init__(text,edges,blosum,variant='H2')
        self.variant=variant
        self.hierarchy=variant!='F4'

    def disease_table(self):
        if self.variant=='F6':return F.gelu(self.text_projection(self.text))
        return super().disease_table()

    def local_features(self,aa,mask):
        onehot=F.one_hot(aa,21).float();blosum=self.blosum[aa]
        if self.variant=='F1':onehot=torch.zeros_like(onehot)
        if self.variant=='F2':blosum=torch.zeros_like(blosum)
        return torch.cat([onehot,blosum],dim=-1)*mask.unsqueeze(-1)

    def forward(self,esm,global_esm,aa,mask,ptm,disease,disease_table=None,return_aux=False):
        d=(self.disease_table() if disease_table is None else disease_table)[disease]
        zero=d.new_zeros((len(aa),31,128))
        if self.variant=='F3':
            # Bypass the ENTIRE local-CNN contribution, including learned biases.
            local=zero
        else:
            descriptors=self.local_features(aa,mask)
            branches=[F.gelu(conv(descriptors.transpose(1,2))).transpose(1,2) for conv in self.local]
            local=self.local_projection(torch.cat(branches,dim=-1))
        protein=d.new_zeros((len(aa),128)) if self.variant=='F4' else self.global_projection(global_esm.float())
        local_esm=zero if self.variant=='F5' else self.esm_projection(esm.float())
        tokens=self.site_norm(local_esm+local+protein.unsqueeze(1)+self.ptm_projection(self.ptm(ptm)).unsqueeze(1)+self.position)
        context,attention=self.attention(d.unsqueeze(1),tokens,tokens,key_padding_mask=~mask,need_weights=return_aux,average_attn_weights=False)
        site=self.condition_norm(tokens[:,15]+context[:,0]);pg=None;sg=None
        if self.hierarchy:
            pg=torch.sigmoid(self.protein_disease_gate(torch.cat([protein,d,protein*d],dim=-1)))
            disease_protein=pg*protein
            sg=torch.sigmoid(self.site_protein_gate(torch.cat([site,disease_protein,d],dim=-1)))
            site=self.hierarchy_norm(site+sg*disease_protein)
        gate=torch.sigmoid(self.gate(torch.cat([site,d,site*d],dim=-1)))
        output=self.head(torch.cat([site*d,gate*site,(1-gate)*d],dim=-1)).squeeze(-1)
        if return_aux:return output,dict(protein_gate=pg,site_gate=sg,residue_attention=attention)
        return output
