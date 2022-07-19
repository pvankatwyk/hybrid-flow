from torch import optim, nn
from nflows import distributions, flows, transforms
from src.utils import get_configs
cfg = get_configs()

class FlowNetwork(nn.Module):
    def __init__(self,):
        super(FlowNetwork, self).__init__()
        self.num_flow_transforms = cfg['model']['flow']['num_flow_transformations']
        self.flow_hidden_features = cfg['model']['flow']['num_flow_transformations']
        self.num_input_features = 1


        # Set up flow
        self.base_dist = distributions.normal.StandardNormal(
            shape=[self.num_input_features],
        )

        t = []
        for _ in range(self.num_flow_transforms):
            t.append(transforms.permutations.RandomPermutation(features=self.num_input_features, ))
            t.append(transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=self.num_input_features,
                hidden_features=self.flow_hidden_features,
                context_features=1
            ))

        self.t = transforms.base.CompositeTransform(t)

        self.flow = flows.base.Flow(
            transform=self.t,
            distribution=self.base_dist
        )

        self.optimizer = optim.Adam(self.flow.parameters())

    def sample(self, num_samples):
        samples_logprobs = self.flow.sample_and_log_prob(num_samples=num_samples,)
        new_samples = samples_logprobs[0].detach().numpy()
        log_probs = samples_logprobs[1].detach().numpy()
        return new_samples, log_probs


class PredictorNetwork(nn.Module):
    def __init__(self):
        super(PredictorNetwork, self).__init__()

        self.input = nn.Linear(1, 64)
        self.linear1 = nn.Linear(64, 32, bias=True)
        self.linear_out = nn.Linear(32, 1)
        # self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear_out(x)
        return x


class Glacier_HybridFlow(nn.Module):
    def __init__(self,):
        super(Glacier_HybridFlow, self).__init__()
        self.Predictor = PredictorNetwork()
        self.Flow = FlowNetwork()

    def forward(self, x):
        z, abslogdet = self.Flow.t(x) # transforms data into noise (latent space)
        pred = self.Predictor(z)
        uncertainty = self.Flow.flow.log_prob(x).exp()
        return pred, uncertainty
