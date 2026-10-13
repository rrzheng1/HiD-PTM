import math

class EarlyStopping:
    def __init__(self,patience=5,min_delta=.0001):
        if patience<1 or min_delta<0:raise ValueError('Invalid early-stopping settings')
        self.patience=patience;self.min_delta=min_delta;self.anchor=-math.inf;self.wait=0

    def update(self,score):
        if not math.isfinite(score):raise ValueError('Nonfinite monitored AP')
        if score>self.anchor+self.min_delta:self.anchor=score;self.wait=0
        else:self.wait+=1
        return self.wait>=self.patience

    def state_dict(self):return dict(anchor=self.anchor,wait=self.wait,patience=self.patience,min_delta=self.min_delta)
    def load_state_dict(self,state):
        if state['patience']!=self.patience or state['min_delta']!=self.min_delta:raise ValueError('Early stopping config changed')
        self.anchor=state['anchor'];self.wait=state['wait']
