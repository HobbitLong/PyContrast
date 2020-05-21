import torch
import torch.nn as nn
import torch.nn.functional as F
from .alias_multinomial import AliasMethod


class BaseMem(nn.Module):
    """Base Memory Class"""
    def __init__(self, K=65536, T=0.07, m=0.5):
        super(BaseMem, self).__init__()
        self.K = K
        self.T = T
        self.m = m

    def _update_memory(self, memory, x, y):
        """
        Args:
          memory: memory buffer
          x: features
          y: index of updating position
        """
        with torch.no_grad():
            x = x.detach()
            w_pos = torch.index_select(memory, 0, y.view(-1))
            w_pos.mul_(self.m)
            w_pos.add_(torch.mul(x, 1 - self.m))
            updated_weight = F.normalize(w_pos)
            memory.index_copy_(0, y, updated_weight)

    def _compute_logit(self, x, w):
        """
        Args:
          x: feat, shape [bsz, n_dim]
          w: softmax weight, shape [bsz, self.K + 1, n_dim]
        """
        x = x.unsqueeze(2)
        out = torch.bmm(w, x)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()
        return out


class RGBMem(BaseMem):
    """Memory bank for single modality"""
    def __init__(self, n_dim, n_data, K=65536, T=0.07, m=0.5):
        super(RGBMem, self).__init__(K, T, m)
        # create sampler
        self.multinomial = AliasMethod(torch.ones(n_data))
        self.multinomial.cuda()

        # create memory bank
        self.register_buffer('memory', torch.randn(n_data, n_dim))
        self.memory = F.normalize(self.memory)

    def forward(self, x, y, x_jig=None, all_x=None, all_y=None):
        """
        Args:
          x: feat on current node
          y: index on current node
          x_jig: jigsaw feat on current node
          all_x: gather of feats across nodes; otherwise use x
          all_y: gather of index across nodes; otherwise use y
        """
        bsz = x.size(0)
        n_dim = x.size(1)

        # sample negative features
        idx = self.multinomial.draw(bsz * (self.K + 1)).view(bsz, -1)
        idx.select(1, 0).copy_(y.data)
        w = torch.index_select(self.memory, 0, idx.view(-1))
        w = w.view(bsz, self.K + 1, n_dim)

        # compute logits
        logits = self._compute_logit(x, w)
        if x_jig is not None:
            logits_jig = self._compute_logit(x_jig, w)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        if (all_x is not None) and (all_y is not None):
            self._update_memory(self.memory, all_x, all_y)
        else:
            self._update_memory(self.memory, x, y)

        if x_jig is not None:
            return logits, logits_jig, labels
        else:
            return logits, labels


class CMCMem(BaseMem):
    """Memory bank for two modalities, e.g. in CMC"""
    def __init__(self, n_dim, n_data, K=65536, T=0.07, m=0.5):
        super(CMCMem, self).__init__(K, T, m)
        # create sampler
        self.multinomial = AliasMethod(torch.ones(n_data))
        self.multinomial.cuda()

        # create memory bank
        self.register_buffer('memory_1', torch.randn(n_data, n_dim))
        self.register_buffer('memory_2', torch.randn(n_data, n_dim))
        self.memory_1 = F.normalize(self.memory_1)
        self.memory_2 = F.normalize(self.memory_2)

    def forward(self, x1, x2, y, x1_jig=None, x2_jig=None,
                all_x1=None, all_x2=None, all_y=None):
        """
        Args:
          x1: feat of modal 1
          x2: feat of modal 2
          y: index on current node
          x1_jig: jigsaw feat of modal1
          x2_jig: jigsaw feat of modal2
          all_x1: gather of feats across nodes; otherwise use x1
          all_x2: gather of feats across nodes; otherwise use x2
          all_y: gather of index across nodes; otherwise use y
        """
        bsz = x1.size(0)
        n_dim = x1.size(1)

        # sample negative features
        idx = self.multinomial.draw(bsz * (self.K + 1)).view(bsz, -1)
        idx.select(1, 0).copy_(y.data)

        w1 = torch.index_select(self.memory_1, 0, idx.view(-1))
        w1 = w1.view(bsz, self.K + 1, n_dim)
        w2 = torch.index_select(self.memory_2, 0, idx.view(-1))
        w2 = w2.view(bsz, self.K + 1, n_dim)

        # compute logits
        logits1 = self._compute_logit(x1, w2)
        logits2 = self._compute_logit(x2, w1)
        if (x1_jig is not None) and (x2_jig is not None):
            logits1_jig = self._compute_logit(x1_jig, w2)
            logits2_jig = self._compute_logit(x2_jig, w1)

        # set label
        labels = torch.zeros(bsz, dtype=torch.long).cuda()

        # update memory
        if (all_x1 is not None) and (all_x2 is not None) \
                and (all_y is not None):
            self._update_memory(self.memory_1, all_x1, all_y)
            self._update_memory(self.memory_2, all_x2, all_y)
        else:
            self._update_memory(self.memory_1, x1, y)
            self._update_memory(self.memory_2, x2, y)

        if (x1_jig is not None) and (x2_jig is not None):
            return logits1, logits2, logits1_jig, logits2_jig, labels
        else:
            return logits1, logits2, labels
