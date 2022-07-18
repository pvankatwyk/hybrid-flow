from src.models.GrIS_HybridFlow import GrIS_HybridFlow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from src.data.FAIR import get_FAIR_data


def train_GrIS_HF(cfg):
    num_epochs = cfg['training']['epochs']
    verbose = cfg['training']['verbose']
    ssp = cfg['data']['ssp']
    region = cfg['training']['epochs']
    batch_size = cfg['training']['batch_size']

    # Data
    inputs, labels, orig_df = get_FAIR_data(
        dir=cfg['data']['dir'],
        ice_source=cfg['data']['ice_source'],
        ssp=cfg['data']['ssp'],
        region=cfg['data']['ice_source'],
    )
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=cfg['training']['test_split_ratio'], )

    # setup optimizers and losses
    HybridFlow = GrIS_HybridFlow()
    optimizer = optim.Adam(list(HybridFlow.Flow.parameters()) + list(HybridFlow.Predictor.parameters()))
    predictor_loss = nn.MSELoss()
    scaling_constant = cfg['training']['generative_scaling_constant']
    loss_list = []
    flow_loss_list = []
    predictor_loss_list = []

    for epoch in range(num_epochs):
        if verbose and epoch % 1 == 0:
            print(f'---------- EPOCH: {epoch} ----------')

        for i in range(0, len(inputs), batch_size):
            x = torch.tensor(X_train[i:i + batch_size, :], dtype=torch.float32)
            y = torch.tensor(y_train[i:i + batch_size], dtype=torch.float32).reshape(-1, 1)

            # Train model
            optimizer.zero_grad()

            # Generative loss
            neg_log_prob = -HybridFlow.Flow.flow.log_prob(inputs=x).mean()

            # Predictor Loss
            pred, uq = HybridFlow(x)
            pred_loss = predictor_loss(pred, y)

            # Cumulative loss (loss = scaling_constant * neg_log_prob + pred_loss)
            loss = torch.add(pred_loss, neg_log_prob, alpha=scaling_constant)
            loss.backward()

            # Keep track of metrics
            loss_list.append(loss.detach().numpy())
            flow_loss_list.append(neg_log_prob.detach().numpy())
            predictor_loss_list.append(pred_loss.detach().numpy())

            optimizer.step()

            if verbose and i % 2000 == 0:
                print(f"Total Loss: {loss}, -Log Prob: {neg_log_prob}, MSE: {pred_loss}")

    if cfg['training']['plot_loss']:
        plt.plot(loss_list, 'r-', label='Total Loss')
        plt.plot(flow_loss_list, 'b-', label='Flow Loss')
        plt.plot(predictor_loss_list, 'g-', label='Predictor Loss')
        plt.title('GrIS_HybridFlow Loss per Batch')
        plt.xlabel(f'Batch # ({batch_size} per batch)')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # Test predictions
    with torch.no_grad():
        predictions, uncertainties = HybridFlow(torch.tensor(X_test, dtype=torch.float32))

    predictions = predictions.numpy().squeeze()
    uncertainties = uncertainties.numpy().squeeze()

    # Calculate metrics
    mae = mean_absolute_error(y_true=y_test, y_pred=predictions)
    pred_loss = mean_squared_error(y_true=y_test, y_pred=predictions, squared=True)
    rmse = mean_squared_error(y_true=y_test, y_pred=predictions, squared=False)

    # Format outputs
    data = {'X_test': X_test, 'y_test': y_test, 'predictions': predictions, 'uncertainties': uncertainties}
    metrics = {f'Test Loss ({predictor_loss})': pred_loss, 'MAE': mae, 'RMSE': rmse}

    return data, metrics
