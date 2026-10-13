import torch
from torch.nn import functional as F
from h2_reference import SiteDiseaseModel as H2

def compact_f3_state_dict(state):
    """Remove only the unused descriptor branch from a saved F3 model."""
    return {k:v for k,v in state.items() if k!='blosum' and not k.startswith(('local.','local_projection.'))}

class SiteDiseaseModel(H2):
    def __init__(self,text,edges,blosum=None,variant='L0'):
        # Preserve common initialization order, then physically remove unused
        # modules. F3 compact weights therefore load with strict=True.
        if blosum is None:blosum=torch.zeros(21,20)
        super().__init__(text,edges,blosum,variant='H2')
        self.variant=variant;self.hierarchy=variant in ['L0','L3']
        del self.local,self.local_projection,self.blosum
        if not self.hierarchy:
            del self.protein_disease_gate,self.site_protein_gate,self.hierarchy_norm
        if variant=='L2':del self.global_projection
        if variant=='L3':del self.gcn,self.graph_norm,self.adjacency

    def disease_table(self):
        if self.variant=='L3':return F.gelu(self.text_projection(self.text))
        return super().disease_table()

    def forward(self,esm,global_esm,aa,mask,ptm,disease,disease_table=None,return_aux=False):
        d=(self.disease_table() if disease_table is None else disease_table)[disease]
        protein=d.new_zeros((len(mask),128)) if self.variant=='L2' else self.global_projection(global_esm.float())
        tokens=self.site_norm(self.esm_projection(esm.float())+protein.unsqueeze(1)+
                              self.ptm_projection(self.ptm(ptm)).unsqueeze(1)+self.position)
        context,attention=self.attention(d.unsqueeze(1),tokens,tokens,key_padding_mask=~mask,need_weights=return_aux,average_attn_weights=False)
        site=self.condition_norm(tokens[:,15]+context[:,0]);pg=None;sg=None
        if self.hierarchy:
            pg=torch.sigmoid(self.protein_disease_gate(torch.cat([protein,d,protein*d],dim=-1)))
            conditional=pg*protein
            sg=torch.sigmoid(self.site_protein_gate(torch.cat([site,conditional,d],dim=-1)))
            site=self.hierarchy_norm(site+sg*conditional)
        gate=torch.sigmoid(self.gate(torch.cat([site,d,site*d],dim=-1)))
        logits=self.head(torch.cat([site*d,gate*site,(1-gate)*d],dim=-1)).squeeze(-1)
        if return_aux:return logits,dict(protein_gate=pg,site_gate=sg,residue_attention=attention)
        return logits
