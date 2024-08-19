import torch
from torch.nn import functional as F


# This is an implementation of Algorithm 3 (Regularised Bayesian Logistic Regression with a Laplace Approximation)
# from "An Empirical Evaluation of Thompson Sampling" by Olivier Chapelle & Lihong Li
# https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf


class PytorchLogisticRegression(torch.nn.Module):
    def __init__(self, n_dim, n_items):
        super(PytorchLogisticRegression, self).__init__()
        self.m = torch.nn.Parameter(torch.Tensor(n_items, n_dim + 1))
        torch.nn.init.normal_(self.m, mean=0.0, std=1.0)
        self.prev_iter_m = self.m.detach().clone()
        self.q = torch.ones((n_items, n_dim + 1))
        self.logloss = torch.nn.BCELoss(reduction="sum")
        self.eval()

    def forward(self, x, sample=False):
        """Predict outcome for all items, allow for posterior sampling"""
        if sample:
            return torch.sigmoid(
                F.linear(
                    x, self.m + torch.normal(mean=0.0, std=1.0 / torch.sqrt(self.q))
                )
            )
        else:
            return torch.sigmoid(F.linear(x, self.m))

    def predict_item(self, x, a):
        """Predict outcome for an item a, only MAP"""
        return torch.sigmoid((x * self.m[a]).sum(axis=1))

    def loss(self, predictions, labels):
        prior_dist = self.q[:, :-1] * (self.prev_iter_m[:, :-1] - self.m[:, :-1]) ** 2
        return 0.5 * prior_dist.sum() + self.logloss(predictions, labels)

    def laplace_approx(self, X, item):
        P = (1 + torch.exp(1 - X.matmul(self.m[item, :].T))) ** (-1)
        self.q[item, :] += (P * (1 - P)).T.matmul(X**2).squeeze(0)

    def update_prior(self):
        self.prev_iter_m = self.m.detach().clone()
