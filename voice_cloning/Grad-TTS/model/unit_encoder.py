from model.text_encoder import *


class UnitEncoder(BaseModule):
    def __init__(self, n_vocab, n_feats, n_channels, filter_channels, 
                 filter_channels_dp, n_heads, n_layers, kernel_size, 
                 p_dropout, window_size=None, n_contentvec=0, spk_emb_dim=64, n_spks=1):
        super(UnitEncoder, self).__init__()
        self.n_vocab = n_vocab
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks

        if n_contentvec > 0:
            self.emb = torch.nn.Linear(n_contentvec, n_channels)
        else:
            self.emb = torch.nn.Embedding(n_vocab, n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, n_channels**-0.5)

        self.prenet = ConvReluNorm(n_channels, n_channels, n_channels, 
                                   kernel_size=5, n_layers=3, p_dropout=0.5)

        self.encoder = Encoder(n_channels + (spk_emb_dim if n_spks > 1 else 0), filter_channels, n_heads, n_layers, 
                               kernel_size, p_dropout, window_size=window_size)

        self.proj_m = torch.nn.Conv1d(n_channels + (spk_emb_dim if n_spks > 1 else 0), n_feats, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.prenet(x, x_mask)
        x = self.encoder(x, x_mask)
        mu_x = self.proj_m(x) * x_mask

        return mu_x, x, x_mask

