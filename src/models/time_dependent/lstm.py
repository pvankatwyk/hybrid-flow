import torch
from torch import nn
from src.data.GrIS import GrIS, GrISDataset
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import plotly.express as px


# GO OVER IMPLEMENTATION AGAIN USING https://www.crosstab.io/articles/time-series-pytorch-lstm

class LSTM(nn.Module):
    def __init__(self, num_input_features, hidden_units):
        super().__init__()
        self.num_input_features = num_input_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=num_input_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


batch_size = 100
divide_year = 2090
epochs = 10
lag = 3
gris = GrIS(local_path=r'C:/Users/Peter/Downloads/climate_time_data.csv')
gris = gris.load()
gris = gris.format(lag=lag, filters={'ice_source': 'GrIS'}, drop_columns=['region', 'collapse'])

train, test = gris.split(type='temporal', divide_year=divide_year, num_samples=5)

target = "SLE"
# target_mean = train[target].mean()
# target_stdev = train[target].std()
#
# for c in train.columns:
#     mean = train[c].mean()
#     stdev = train[c].std()
#
#     train[c] = (train[c] - mean) / stdev
#     test[c] = (test[c] - mean) / stdev

X_train = torch.tensor(np.array(train.drop(columns=['SLE'])), dtype=torch.float32)
y_train = torch.tensor(np.array(train['SLE']), dtype=torch.float32)
X_test = torch.tensor(np.array(test.drop(columns=['SLE'])), dtype=torch.float32)
y_test = torch.tensor(np.array(test['SLE']), dtype=torch.float32)

train_dataset = GrISDataset(X_train, y_train, sequence_length=5)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = GrISDataset(X_test, y_test, sequence_length=5)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X, y = next(iter(train_loader))
print("Features shape:", X.shape)
print("Target shape:", y.shape)

model = LSTM(num_input_features=X.shape[-1], hidden_units=100)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")


print("Untrained test\n--------")
test_model(test_loader, model, loss_function)
print()

for ix_epoch in range(epochs):
    print(f"Epoch {ix_epoch+1}\n---------")
    train_model(train_loader, model, loss_function, optimizer=optimizer)
    test_model(test_loader, model, loss_function)
    print()


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output


train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

ystar_col = "Predicted SLE"
train[ystar_col] = predict(train_eval_loader, model).numpy()
test[ystar_col] = predict(test_loader, model).numpy()

df_out = pd.concat((train, test))

# for c in df_out.columns:
#     df_out[c] = df_out[c] * target_stdev + target_mean

print(df_out.head())
sample1_ssp119 = df_out[(df_out['sample'] == 1) & (df_out.ssp == 119)]
sample1_ssp119.index = sample1_ssp119['year']
sample1_ssp119 = sample1_ssp119[['SLE', 'Predicted SLE']]


fig = px.line(sample1_ssp119, labels=dict(created_at="Date", value="Sea Level Contribution"))
fig.add_vline(x=divide_year, line_width=4, line_dash="dash")
fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text="Test set start", showarrow=False)
# fig.update_layout(
#     template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
# )
fig.write_html('sle.html', auto_open=True)

stop = ''