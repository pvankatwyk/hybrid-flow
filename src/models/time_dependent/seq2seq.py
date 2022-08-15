import torch
from src.data.GrIS import GrIS


gris_data = GrIS(local_path=r'C:/Users/Peter/Downloads/climate_time_data.csv')
gris_data = gris_data.load()
gris_data = gris_data.format(lag=5, filters={'ice_source': 'GrIS'}, drop_columns=['region', 'collapse'])


class Encoder(torch.nn.Module):
    def __init__(self, num_rows, num_features=10, num_rnn_layers=1, hidden_size=100, rnn_dropout=0.2):
        super().__init__()
        self.sequence_length = num_rows
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.device = 'cpu'
        self.gru = torch.nn.GRU(
            num_layers=num_rnn_layers,
            input_size=self.num_features,
            batch_first=True,
            bidirectional=False,
            dropout=rnn_dropout,
        )

    def forward(self, input):
        ht = torch.zeros(self.num_features, input.size(0), self.hidden_size, device=self.device)
        gru_out, hidden = self.gru(input, ht)
        if self.num_features > 1:
            hidden = hidden.view(input.size(0), self.sequence_length, 1, self.hidden_size)
            if self.num_features > 0:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
            hidden = hidden.sum(axis=0)
        return gru_out, hidden


class Decoder(torch.nn.Module):
    def __init__(self, input_length, hidden_size, dropout=0.2):
        super().__init__()
        self.decoder_rnn = torch.nn.GRUCell(
            input_size=input_length,
            hidden_size=hidden_size,
        )
        self.out = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, prev_hidden, y):
        rnn_hidden = self.decoder_rnn(y, prev_hidden)
        output = self.out(rnn_hidden)
        return output, self.dropout(rnn_hidden)


class Seq2SeqModel(torch.nn.Module):
    def __init__(self, encoder, decoder_cell, output_size=5, teacher_forcing=0.3, sequence_len=336, decoder_input=True, device='cpu'):
        super().__init__()
        self.encoder = encoder
        self.decoder_cell = decoder_cell
        self.output_size = output_size
        self.teacher_forcing = teacher_forcing
        self.sequence_length = sequence_len
        self.decoder_input = decoder_input
        self.device = device

    def forward(self, xb, yb=None):
        if self.decoder_input:
            decoder_input = xb[-1]
            input_seq = xb[0]
            if len(xb) > 2:
                encoder_output, encoder_hidden = self.encoder(input_seq, *xb[1:-1])
            else:
                encoder_output, encoder_hidden = self.encoder(input_seq)
        else:
            if type(xb) is list and len(xb) > 1:
                input_seq = xb[0]
                encoder_output, encoder_hidden = self.encoder(*xb)
            else:
                input_seq = xb
                encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden
        outputs = torch.zeros(input_seq.size(0), self.output_size, device=self.device)
        y_prev = input_seq[:, -1, 0].unsqueeze(1)
        for i in range(self.output_size):
            step_decoder_input = torch.cat((y_prev, decoder_input[:, i]), axis=1)
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                step_decoder_input = torch.cat((yb[:, i].unsqueeze(1), decoder_input[:, i]), axis=1)
            rnn_output, prev_hidden = self.decoder_cell(prev_hidden, step_decoder_input)
            y_prev = rnn_output
            outputs[:, i] = rnn_output.squeeze(1)
        return outputs


stop = ''
