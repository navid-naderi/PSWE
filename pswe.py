import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinterp1d import Interp1d

class PSWE(nn.Module):
    def __init__(self, d_in, num_ref_points, num_projections):
        '''
        The PSWE module that produces fixed-dimensional permutation-invariant embeddings for input sets of arbitrary size.
        Inputs:
            d_in:  The dimensionality of the space that each set sample belongs to
            num_ref_points: Number of points in the reference set
            num_projections: Number of slices
        '''
        super(PSWE, self).__init__()
        self.d_in = d_in
        self.num_ref_points = num_ref_points
        self.num_projections = num_projections
        self.ref = ref

        uniform_ref = torch.linspace(-1, 1, num_ref_points).unsqueeze(1).repeat(1, num_projections)
        self.reference = nn.Parameter(uniform_ref)

        # slicer
        self.theta = nn.utils.weight_norm(nn.Linear(d_in, num_projections, bias=False), dim=0)
        if num_projections <= d_in:
            nn.init.eye_(self.theta.weight_v)
        else:
            nn.init.normal_(self.theta.weight_v)
        self.theta.weight_g.data = torch.ones_like(self.theta.weight_g.data, requires_grad=False)
        self.theta.weight_g.requires_grad = False

        # weights to reduce the output embedding dimensionality
        self.weight = nn.Parameter(torch.zeros(num_projections, num_ref_points))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        Output:
            weighted_embeddings: B x num_projections tensor, containing a batch of B embeddings, each of dimension "num_projections" (i.e., number of slices)
        '''

        B, N, dn = X.shape
        Xslices = self.get_slice(X)
        Xslices_sorted, Xind = torch.sort(Xslices, dim=1)


        M, dm = self.reference.shape

        if M == N:
            Xslices_sorted_interpolated = Xslices_sorted
        else:
            x = torch.linspace(0, 1, N + 2)[1:-1].unsqueeze(0).repeat(B * self.num_projections, 1).to(X.device)
            xnew = torch.linspace(0, 1, M + 2)[1:-1].unsqueeze(0).repeat(B * self.num_projections, 1).to(X.device)
            y = torch.transpose(Xslices_sorted, 1, 2).reshape(B * self.num_projections, -1)
            Xslices_sorted_interpolated = torch.transpose(Interp1d()(x, y, xnew).view(B, self.num_projections, -1), 1, 2)

        Rslices = self.reference.expand(Xslices_sorted_interpolated.shape)

        _, Rind = torch.sort(Rslices, dim=1)
        embeddings = (Rslices - torch.gather(Xslices_sorted_interpolated, dim=1, index=Rind)).permute(0, 2, 1)

        w = self.weight.unsqueeze(0).repeat(B, 1, 1)
        weighted_embeddings = (w * embeddings).sum(-1)
        return weighted_embeddings


    def get_slice(self, X):
        '''
        Slices samples from distribution X~P_X
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        '''
        return self.theta(X)
