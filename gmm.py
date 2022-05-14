import torch
import math
import pytorch_lightning as pl


class GaussianMixtureEstimator(torch.nn.Module):
    def __init__(self, dims, n_components):
        super().__init__()
        self.dims = dims
        self.n_components = n_components
        self.mu = torch.nn.parameter.Parameter(torch.randn(
            (self.n_components, self.dims), dtype=torch.float), requires_grad=True)
        self.logvars = torch.nn.parameter.Parameter(torch.zeros(
            (self.n_components, self.dims), dtype=torch.float), requires_grad=True)
        self.logpriors = torch.nn.parameter.Parameter(torch.zeros(
            (self.n_components), dtype=torch.float), requires_grad=True)
        self.pi = torch.Tensor([math.pi])

    def log_gaussian(self, x, mean=0, logvar=0.):
        """
        Returns the density of x under the supplied gaussian. Defaults to
        standard gaussian N(0, I)
        :param x: (*) torch.Tensor
        :param mean: float or torch.FloatTensor with dimensions (*)
        :param logvar: float or torch.FloatTensor with dimensions (*)
        :return: (*) elementwise log density
        """

        a = (x - mean) ** 2
        log_p = -0.5 * (logvar + a / logvar.exp())
        log_p = log_p - 0.5 * torch.log(2 * self.pi)

        return log_p

    def get_likelihoods(self, X, mu, logvar):
        """
        :param X: design matrix (examples, features)
        :param mu: the component means (K, features)
        :param logvar: the component log-variances (K, features)
            Note: exponentiating can be unstable in high dimensions.
        :return likelihoods: (K, examples)
        """

        # get feature-wise log-likelihoods (K, examples, features)
        log_likelihoods = self.log_gaussian(
            X[None, :, :],  # (1, examples, features)
            mu[:, None, :],  # (K, 1, features)
            logvar[:, None, :]  # (K, 1, features)
        )

        # sum over the feature dimension
        log_likelihoods = log_likelihoods.sum(-1)

        return log_likelihoods

    def get_log_posteriors(self, log_likelihoods, log_priors):
        """
        Calculate the the posterior probabilities log p(z|x)
        :param log_likelihoods: the log relative likelihood log p(x|z), of each data point under each mode (K, examples)
        :param log_priors: the log priors log p(z), of each mode (K)
        :return: the log posterior p(z|x) (K, examples)
        """
        # self.logpriors is not normalized such that sum(p)=1
        norm_logpriors = log_priors - log_priors.exp().sum().log()
        # include priors
        weighted_log_likelihood = log_likelihoods + norm_logpriors
        # compute posterior
        log_posteriors = weighted_log_likelihood - \
            weighted_log_likelihood.exp().sum(0).log()

        return log_posteriors

    def forward(self, inputs):
        log_likelihoods = self.get_likelihoods(inputs, self.mu, self.logvars)
        return self.get_log_posteriors(log_likelihoods, self.logpriors)


class ClusteringTrain(pl.LightningModule):
    def __init__(self, n_components, dim, preprocessor=None) -> None:
        super().__init__()
        self.gmm = GaussianMixtureEstimator(dim, n_components)
        self.preprocessor = torch.nn.Sequential() if preprocessor is None else preprocessor

    def forward(self, inputs):
        self.gmm(self.preprocessor(inputs))

    def training_step(self, batch, batch_idx):
        log_posteriors: torch.Tensor = self(batch)
        loss = - log_posteriors.sum()
        self.log("train loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        log_posteriors: torch.Tensor = self(batch)
        loss = - log_posteriors.sum()
        self.log("test loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        log_posteriors: torch.Tensor = self(batch)
        loss = - log_posteriors.sum()
        self.log("test loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.gmm.parameters(), lr=0.001)
