import torch
import torch.nn as nn
from torch.nn import Linear


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_head = config.get('num_head', 8)
        n_layers = 9
        channels = config['conv_channels']
        out_dim = config['out_dim']
        dropout = config['dropout']
        n_encoder_inputs = n_decoder_inputs = config["var_len"]
        self.output_len = config['output_len']

        self.dropout = dropout

        self.input_pos_embedding = torch.nn.Embedding(256, embedding_dim=channels)
        self.target_pos_embedding = torch.nn.Embedding(256, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_head,
            dropout=self.dropout,
            dim_feedforward=4 * channels,

        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=channels,
            nhead=8,
            dropout=self.dropout,
            dim_feedforward=4 * channels,
        )  ###

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)  ###

        self.input_projection = nn.Conv1d(n_encoder_inputs, channels, kernel_size=3)
        self.output_projection = Linear(n_decoder_inputs, channels)  ###

        self.linear = Linear(channels, 4)  ###

        self.do = nn.Dropout(p=self.dropout)
        self.end_conv = nn.Conv1d(channels, out_dim, kernel_size=3)

    def encode_src(self, src):  # (batch_size, input_window, num_nodes)
        src = torch.permute(src, dims=[0, 2, 1])
        src = torch.unsqueeze(src, dim=1)  # (batch_size, 1, num_nodes, input_window)
        src_start = self.input_projection(src).permute(0, 2, 3, 1).contiguous()  # (batch_size, num_nodes, input_window, D)

        batch_size, id_len, in_sequence_len, var_len = src_start.shape

        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0).unsqueeze(0)
            .repeat(batch_size, id_len, 1)
        )

        pos_encoder = self.input_pos_embedding(pos_encoder)

        # src = src_start + pos_encoder
        src = pos_encoder
        src = src.permute(2, 0, 1, 3)  # (input_window, batch_size, num_nodes, D)
        src = src.reshape(in_sequence_len, batch_size * id_len, var_len).contiguous()
        src = self.encoder(src).reshape(in_sequence_len, batch_size, id_len, -1).permute(1, 2, 0, 3) + src_start  # (batch_size, num_nodes, input_window, D)

        return src.permute(0, 3, 1, 2)  # (batch_size, D, num_nodes, input_window)

    def decode_trg(self, trg, memory):
        trg_start = self.output_projection(trg).permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start

        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        out = self.decoder(tgt=trg, memory=memory, tgt_mask=trg_mask) + trg_start

        out = out.permute(1, 0, 2)

        out = self.linear(out)

        return out

    def forward(self, src, trg):

        src = self.encode_src(src)

        out = self.decode_trg(trg=trg, memory=src)

        return out

    # def forward(self, src):
    #     out = self.encode_src(src)
    #     out = self.end_conv(out)
    #     return out[..., -self.output_len:], None
