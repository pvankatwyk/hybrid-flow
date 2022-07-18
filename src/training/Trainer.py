from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from src.data.FAIR import get_FAIR_data


class Trainer:
    def __init__(self, cfg):
        self.HybridFlow = None
        self.data = {}
        self.cfg = cfg
        self.model_name = cfg['training']['model']
        self.num_epochs = cfg['training']['epochs']
        self.verbose = cfg['training']['verbose']
        self.batch_size = cfg['training']['batch_size']
        self.num_input_features = None
        self.loss_logs = {'total_loss': [], 'flow_loss': [], 'predictor_loss': []}

    def load_data(self):
        inputs, labels, orig_df = get_FAIR_data(
            dir=self.cfg['data']['dir'],
            ice_source=self.cfg['data']['ice_source'],
            ssp=self.cfg['data']['ssp'],
            region=self.cfg['data']['region'],
        )
        X_train, X_test, y_train, y_test = train_test_split(inputs, labels,
                                                            test_size=self.cfg['training']['test_split_ratio'], )
        self.data['X_train'] = X_train
        self.data['X_test'] = X_test
        self.data['y_train'] = y_train
        self.data['y_test'] = y_test

        return self

    def train(self):

        # Load data if it hasn't been loaded already
        if len(self.data) == 0:
            self.load_data()

        # setup model type, optimizers and losses
        if self.model_name == 'GrIS_HybridFlow':
            from src.models.GrIS_HybridFlow import GrIS_HybridFlow
            self.HybridFlow = GrIS_HybridFlow()
        elif self.model_name == 'Glacier_HybridFlow':
            from src.models.Glacier_HybridFlow import Glacier_HybridFlow
            self.HybridFlow = Glacier_HybridFlow()
        elif self.model_name == 'AIS_HybridFlow':
            from src.models.AIS_HybridFlow import AIS_HybridFlow
            self.HybridFlow = AIS_HybridFlow()
        else:
            raise NameError(f'Model: {self.model_name} either does not exist or is not supported yet.')

        self.num_input_features = self.HybridFlow.Flow.num_input_features
        optimizer = optim.Adam(list(self.HybridFlow.Flow.parameters()) + list(self.HybridFlow.Predictor.parameters()))
        predictor_loss = nn.L1Loss()
        scaling_constant = self.cfg['training']['generative_scaling_constant']

        for epoch in range(self.num_epochs):
            if self.verbose and epoch % 1 == 0:
                print(f'---------- EPOCH: {epoch} ----------')

            for i in range(0, len(self.data['X_train']), self.batch_size):
                if self.num_input_features == 1:
                    x = torch.tensor(self.data['X_train'][i:i + self.batch_size], dtype=torch.float32).reshape(-1, 1)
                else:
                    x = torch.tensor(self.data['X_train'][i:i + self.batch_size, :], dtype=torch.float32)
                y = torch.tensor(self.data['y_train'][i:i + self.batch_size], dtype=torch.float32).reshape(-1, 1)

                # Train model
                optimizer.zero_grad()

                # Generative loss
                neg_log_prob = -self.HybridFlow.Flow.flow.log_prob(inputs=x).mean()

                # Predictor Loss
                pred, uq = self.HybridFlow(x)
                pred_loss = predictor_loss(pred, y)

                # Cumulative loss (loss = scaling_constant * neg_log_prob + pred_loss)
                loss = torch.add(pred_loss, neg_log_prob, alpha=scaling_constant)
                loss.backward()

                # Keep track of metrics
                self.loss_logs['total_loss'].append(loss.detach().numpy())
                self.loss_logs['flow_loss'].append(neg_log_prob.detach().numpy())
                self.loss_logs['predictor_loss'].append(pred_loss.detach().numpy())

                optimizer.step()

                if self.verbose and i % 2000 == 0:
                    print(f"Total Loss: {loss}, -Log Prob: {neg_log_prob}, MSE: {pred_loss}")

        return self

    def plot_loss(self):
        plt.plot(self.loss_logs['total_loss'], 'r-', label='Total Loss')
        plt.plot(self.loss_logs['flow_loss'], 'b-', label='Flow Loss')
        plt.plot(self.loss_logs['predictor_loss'], 'g-', label='Predictor Loss')
        plt.title('GrIS_HybridFlow Loss per Batch')
        plt.xlabel(f'Batch # ({self.batch_size} per batch)')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def evaluate(self):
        # Test predictions
        self.HybridFlow.eval()

        if self.num_input_features == 1:
            X_test = torch.tensor(self.data['X_test'], dtype=torch.float32).reshape(-1, 1)
        else:
            X_test = torch.tensor(self.data['X_test'], dtype=torch.float32)

        with torch.no_grad():
            predictions, uncertainties = self.HybridFlow(X_test)

        predictions = predictions.numpy().squeeze()
        uncertainties = uncertainties.numpy().squeeze()

        # Calculate metrics
        mae = mean_absolute_error(y_true=self.data['y_test'], y_pred=predictions)
        pred_loss = mean_squared_error(y_true=self.data['y_test'], y_pred=predictions, squared=True)
        rmse = mean_squared_error(y_true=self.data['y_test'], y_pred=predictions, squared=False)

        # Format outputs
        data = {'X_test': self.data['X_test'], 'y_test': self.data['y_test'], 'predictions': predictions,
                'uncertainties': uncertainties}
        metrics = {f'Test Loss ({self.loss_logs["total_loss"][-1]})': pred_loss, 'MAE': mae, 'RMSE': rmse}

        return data, metrics
