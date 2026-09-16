import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation='relu', dropout=0.2):
        super(EncoderLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation

        self.linear = nn.Linear(input_dim, output_dim)

        self.batch_norm = nn.BatchNorm1d(output_dim)

        # Choose activation function
        if self.activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif self.activation == 'leakyrelu':
            self.activation_fn = nn.LeakyReLU(0.2, inplace=True)
        elif self.activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif self.activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError('Unknown activation type %s' % self.activation)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.dropout_layer = nn.Identity()

    def forward(self, x, active=True, batchnorm=True, dropout=True):
        x = self.linear(x)
        if batchnorm:
            x = self.batch_norm(x)
        if active:
            x = self.activation_fn(x)
        if dropout:
            x = self.dropout_layer(x)

        return x


class Encoder(nn.Module):
    """Output latent representation using multiple EncoderLayers"""

    def __init__(self, encoder_dim, activation='relu', dropout=0.2):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.encoder_dims = len(encoder_dim) - 1
        for i in range(self.encoder_dims):
            self.encoder_layers.append(EncoderLayer(encoder_dim[i], encoder_dim[i + 1],
                                                    activation=activation, dropout=dropout
                                                    ))

    def forward(self, x):
        x = self.encoder_layers[0](x, dropout=True)

        for layer in self.encoder_layers[1:-1]:
            x = x + layer(x, dropout=True)

        x = self.encoder_layers[-1](x, dropout=False)

        return x


class MultiLabelClassifier(nn.Module):
    """Shared multi-label classifier C for individual views and primary fusion."""
    def __init__(self, latent_dim, n_classifications):
        super(MultiLabelClassifier, self).__init__()

        self._latent_dim = latent_dim
        self._n_classifications = n_classifications

        self.classification = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),

            nn.Linear(self._latent_dim, self._n_classifications),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.classification(x)
        return y


class CrossViewProjectionHead(nn.Module):
    """View-specific projection head p_v for cross-view contrastive learning."""
    def __init__(self, latent_dim):
        super(CrossViewProjectionHead, self).__init__()
        self._latent_dim = latent_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim, bias=True),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
            nn.Linear(self._latent_dim, self._latent_dim, bias=False),
            nn.BatchNorm1d(self._latent_dim),
        )

    def forward(self, x):
        return self.instance_projector(x)


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation='relu', dropout=0.2):
        super(DecoderLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation

        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)

        # Choose activation function
        if self.activation == 'sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif self.activation == 'leakyrelu':
            self.activation_fn = nn.LeakyReLU(0.2, inplace=True)
        elif self.activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif self.activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError('Unknown activation type %s' % self.activation)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.dropout_layer = nn.Identity()

    def forward(self, x, active=True, batchnorm=True, dropout=True):
        x = self.linear(x)
        if batchnorm:
            x = self.batch_norm(x)
        if active:
            x = self.activation_fn(x)
        if dropout:
            x = self.dropout_layer(x)

        return x


class Decoder(nn.Module):
    """Reconstruct original view from latent representation using multiple DecoderLayers"""

    def __init__(self, decoder_dim, activation='relu', dropout=0.2):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList()
        self.decoder_dims = len(decoder_dim) - 1
        for i in range(self.decoder_dims):
            self.decoder_layers.append(DecoderLayer(decoder_dim[i], decoder_dim[i + 1],
                                                    activation=activation, dropout=dropout
                                                    ))

    def forward(self, x):
        x = self.decoder_layers[0](x, dropout=True)

        for layer in self.decoder_layers[1:-1]:
            x = x + layer(x, dropout=True)  # Residual connection

        x = self.decoder_layers[-1](x, dropout=False)

        return x
