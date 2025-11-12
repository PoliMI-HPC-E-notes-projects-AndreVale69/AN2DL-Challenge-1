"""
PainTCNBiLSTMAttn Model.
"""
from torch import cat, sigmoid, zeros_like, rand, Tensor
from torch.nn import Module, ModuleList, Embedding, Linear, LSTM, Sequential, LayerNorm, RReLU, Dropout

from internal.nn.blocks.attention_pool import AttentionPool
from internal.nn.blocks.squeeze_and_excitation import SE1D
from internal.nn.blocks.static_multi_layer_perceptron import StaticMLP
from internal.nn.blocks.temporal_convolutional_network import TCN


class PainTCNBiLSTMAttn(Module):
    """
    PainTCNBiLSTMAttn Model.

    This model combines Temporal Convolutional Networks (TCN), Bidirectional Long Short-Term Memory (BiLSTM),
    and Attention Pooling to process time-series data for pain prediction tasks. It also incorporates
    static features and summary features through dedicated MLP branches.
    """
    def __init__(self,
                 d_num: int = 90,
                 d_emb: int = 2,
                 d_sta: int = 7,
                 d_summ: int = 0,
                 tcn_channels: int = 128,
                 lstm_hidden: int = 128,
                 num_classes: int = 3,
                 dropout: float = 0.3,
                 p_drop_summ=0.5):
        """
        PainTCNBiLSTMAttn Model.

        :param d_num: Dimension of numerical features. Default is 90.
        :param d_emb: Dimension of survey embeddings. Default is 2.
        :param d_sta: Dimension of static features. Default is 7.
        :param d_summ: Dimension of summary features. Default is 0.
        :param tcn_channels: Number of channels in the TCN layers. Default is 128.
        :param lstm_hidden: Hidden dimension of the LSTM layer. Default is 128.
        :param num_classes: Number of output classes for classification. Default is 3.
        :param dropout: Dropout rate applied in various layers. Default is 0.3.
        :param p_drop_summ: Probability of dropping the summary features branch during training. Default is 0.5.
        """
        super().__init__()

        self.se = SE1D(tcn_channels, reduction_ratio=8)
        """
        **Squeeze-and-Excitation (SE) block for 1D inputs.**
        
        This block applies channel-wise attention to the input tensor by first squeezing
        the input along the temporal dimension to create a channel descriptor, then passing
        this descriptor through a two-layer fully connected network to learn channel-wise weights,
        and finally scaling the original input tensor by these weights.
        
        Summary:
            - Input dimension: tcn_channels
            - Output dimension: tcn_channels
            - Reduction ratio: 8 (used to reduce the channel dimension in the first layer of the network)
        """

        self.survey_embs = ModuleList([Embedding(3, d_emb) for _ in range(4)])
        """
        **Embedding layers for pain surveys.**
        
        Embedding layers for the 4 surveys (pain_survey_1 to pain_survey_4).
        Each survey has 3 levels (0, 1, 2) and is embedded into a vector of dimension `d_emb`.
        This allows the model to learn a dense representation for each survey level.
        
        For each survey:
            - Input dimension: 3 (number of levels)
            - Output dimension: d_emb (embedding dimension)
        4 such embedding layers are created, one for each survey.
        
        For example, if d_emb=2, then each survey level will be represented as a 2-dimensional vector:
            - Level 0 -> [e0_1, e0_2]
            - Level 1 -> [e1_1, e1_2]
            - Level 2 -> [e2_1, e2_2]
        4 surveys will produce 4 such embeddings, which are then concatenated with the numerical features.
        
        Summary:
            - Number of surveys: 4
            - Input dimension per survey: 3
            - Input total dimension from all surveys: surveys * input_dim = 4 * 3 = 12
            - Output dimension per survey: d_emb (default 2)
            - Total embedding dimension from all surveys: 4 * d_emb (default 4 * 2 = 8)
        """

        self.proj = Linear(d_num + 4 * d_emb, tcn_channels)
        """
        **Projection layer to TCN channels.**
        
        Linear layer to project the concatenated input features at each timestep
        (numerical features + survey embeddings) to the desired number of TCN channels.
        
        The concatenated input dimension is d_num + 4 * d_emb, where:
            - d_num: Dimension of numerical features
            - 4 * d_emb: Dimension from the 4 survey embeddings (each of size d_emb, pain_survey_1 to pain_survey_4)    
        
        Summary:
            - Input dimension: d_num + 4 * d_emb
            - Output dimension: tcn_channels
        """

        self.tcn = TCN(channels=tcn_channels, dilation_rates=(1,2,4,8), kernel_size=5, dropout=dropout)
        """
        **Temporal Convolutional Network (TCN) block.**
        
        The TCN consists of a series of Conv1dBlock layers with increasing dilation rates.
        This architecture allows the network to capture long-range dependencies in sequential data.
        
        Details:
            * The kernel size means each Conv1dBlock layer uses a convolutional kernel of size 5.
            * The dilation rates (1, 2, 4, 8) allow the network to have a larger receptive field,
              enabling it to model long-term dependencies in sequential data.
            * Dropout is applied in each Conv1dBlock layer to prevent overfitting.
        
        Summary:
            - Input dimension: tcn_channels
            - Output dimension: tcn_channels
            - Dilation rates: (1, 2, 4, 8) for each Conv1dBlock layer
            - Kernel size: 5
            - Dropout: dropout rate applied in each Conv1dBlock layer (default 0.3)
        """

        self.lstm = LSTM(
            input_size=tcn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        """
        **Bidirectional Long Short-Term Memory (BiLSTM) layer.**
        
        This layer processes the output from the TCN to capture temporal dependencies
        in both forward and backward directions.
        
        `Why LSTM after TCN?` LSTMs are designed to capture longer-term dependencies in sequential data.
        By combining TCNs with LSTMs, the model can leverage the strengths of both architectures:
        (1) TCNs for local feature extraction; (2) LSTMs for modeling long-range temporal dependencies.
        
        `Why bidirectional?` The bidirectional configuration allows the LSTM to access information
        from both past and future contexts simultaneously. So we give the freedom to the model
        to decide which direction is more informative for the task at hand.
        
        `Why only 1 layer?` Because before the LSTM, we already have a powerful TCN feature extractor. 
        It can already capture complex temporal patterns, so stacking multiple LSTM layers
        may lead to diminishing returns and increased computational complexity.
        Also, we want to avoid overfitting, especially if the training data is limited.

        `Why no dropout?` The answer is similar to why we use only 1 layer. Before the LSTM, we have a TCN
        that already includes dropout for regularization. Applying dropout in the LSTM may not be necessary and could
        potentially disrupt the temporal dependencies the LSTM is trying to learn.
        
        Summary:
            - Input dimension: tcn_channels
            - Hidden dimension: lstm_hidden (per direction)
            - Output dimension: 2 * lstm_hidden (due to bidirectionality)
            - Number of layers: 1
            - Batch first: True (input shape is (batch_size, seq_length, features))
            - Dropout: 0.0 (no dropout applied in LSTM)
        """

        self.attn = AttentionPool(in_features=2*lstm_hidden, out_hidden=128)
        """
        **Attention Pooling layer.**
        
        This layer applies attention mechanism to pool the LSTM outputs across the temporal dimension.
        It computes a weighted sum of the LSTM outputs, where the weights are learned based on
        the relevance of each timestep's output to the overall sequence representation.
        
        We choose attention pooling here to allow the model to focus on the most informative
        parts of the sequence when generating a fixed-size representation. This is particularly useful
        in scenarios where certain timesteps may carry more significance for the prediction task.
        
        We choose an attention hidden dimension of 128 as a balance between model capacity
        and computational efficiency. A larger hidden dimension may allow the model to capture
        more complex relationships, but it also increases the number of parameters and computational cost.
        
        Summary:
            - Input dimension: 2 * lstm_hidden
            - Output dimension: 2 * lstm_hidden
            - Attention hidden dimension: 128
        """

        self.static_mlp = StaticMLP(in_dim=d_sta, hidden=64, dropout=dropout)
        """
        **Static Multi-Layer Perceptron (MLP) Block.**
        
        This block processes static input features of dimension `d_sta` and outputs a feature vector of dimension 64.
        It consists of two fully connected layers with RReLU activation and Dropout in between.
        
        We use a static MLP to effectively capture and transform the static features
        into a higher-level representation that can be fused with the dynamic features.
        The choice of 64 as the output dimension provides a good balance between
        model capacity and computational efficiency.
        
        Summary:
            - Input dimension: d_sta
            - Output dimension: 64
            - Hidden dimension: 64
            - Dropout: dropout rate applied after each activation (default 0.3)
        """

        # The number of summary features is variable; if 0, we skip the branch otherwise use MLP
        self.summ_out = summ_out = 16
        self.summ_mlp: Sequential = Sequential(
            # First layer transforms input features to hidden dimension
            LayerNorm(d_summ),
            # Second layer outputs a fixed-size summary feature vector
            Linear(d_summ, 128), RReLU(0.1, 0.3), Dropout(dropout),
            # Third layer maintains hidden dimension but at the same time reduces to summ_out
            Linear(128, summ_out), RReLU(0.1, 0.3)
        )
        """
        **Summary Features MLP Block.**
        
        This block processes summary input features of dimension `d_summ` and
        outputs a feature vector of dimension `summ_out`.
        It consists of a LayerNorm followed by two fully connected layers with RReLU activation and Dropout in between.
        
        We use this MLP to transform the summary features into a higher-level representation
        that can be fused with the dynamic and static features. The choice of `summ_out` as 16 provides
        a compact yet informative representation of the summary features.
        
        Summary:
            - Input dimension: d_summ
            - Output dimension: summ_out (16)
            - Hidden dimension: 128
            - Dropout: dropout rate applied after each activation (default 0.3)
        """

        self.p_drop_summ = p_drop_summ
        """
        **Probability of dropping the summary features branch during training.**
        
        This parameter controls the likelihood of dropping the entire summary features branch
        during each training step. Dropping this branch encourages the model to rely more on
        the dynamic and static features, preventing it from becoming overly dependent on the summary features.
        
        This technique has been developed since summary features can sometimes provide
        an easy shortcut for the model to make predictions, which may lead to overfitting.
        We have noticed that by occasionally dropping this branch during training,
        the model learns to better utilize the dynamic and static features, resulting in improved generalization.
        
        For example, with p_drop_summ=0.5, there is a 50% chance that the summary features
        will be set to zero during each training step. So during training, the model will
        see a mix of inputs with and without summary features, helping it to learn robust representations.
        
        If set to 0.0, the summary features branch will always be used. If set to 1.0, the summary features branch will
        always be dropped during training.
        
        Summary:
            - Value range: [0.0, 1.0]
            - Default: 0.5
            - Input dimension: N/A
            - Output dimension: N/A
        """

        self.gate_fc = Linear(64 + summ_out, 2*lstm_hidden)
        """
        **Gating mechanism for dynamic features.**
        
        This linear layer is like an hardware gate that controls how much of the dynamic feature vector
        should be allowed to pass through based on the context provided by the static and summary features.
        
        It takes as input the concatenated static and summary feature vectors (of dimension 64 + summ_out)
        and outputs a gate vector of dimension 2 * lstm_hidden (same as the dynamic feature vector).
        The output values are then passed through a sigmoid activation to obtain values between 0 and 1.
        
        The goal of this gating mechanism is to allow the model to adaptively modulate
        the influence of the dynamic features based on the context provided by the static and summary features.
        This should help the model to focus on the most relevant information for making predictions.
        
        The adoption of this gating mechanism was motivated by the observation that
        in some cases, the dynamic features alone may not be sufficient for accurate predictions.
        By conditioning the dynamic features on the static and summary features, the model can
        better capture the interactions between these different types of information.
        
        Summary:
            - Input dimension: 64 + summ_out
            - Output dimension: 2 * lstm_hidden
        """

        self.head = Sequential(
            # First layer transforms concatenated features to hidden dimension
            Linear((2*lstm_hidden) + 64 + summ_out, 128),
            # Activation and dropout
            RReLU(0.1, 0.3), Dropout(dropout),
            # Final layer outputs logits for each class
            Linear(128, num_classes)
        )
        """
        **Final classification head.**
        
        This sequential block serves as the final classification head of the model.
        It takes as input the concatenated feature vector consisting of:
            - Dynamic features (after gating): 2 * lstm_hidden
            - Static features: 64
            - Summary features: summ_out

        The total input dimension is therefore (2 * lstm_hidden) + 64 + summ_out.
        
        The head consists of two fully connected layers with RReLU activation and Dropout in between.
        The first layer transforms the concatenated features to a hidden dimension of 128,
        while the second layer outputs logits for each class (num_classes).
        
        Summary:
            - Input dimension: (2 * lstm_hidden) + 64 + summ_out
            - Hidden dimension: 128
            - Output dimension: num_classes
        """

    def forward(self, x_num: Tensor, x_surv_list: list[Tensor], x_sta: Tensor, x_summ: Tensor) -> Tensor:
        """
        Forward pass of the PainTCNBiLSTMAttn model.

        The model processes numerical time-series data, survey embeddings, static features,
        and summary features to produce logits for classification.

        1. Embedding Layer: Each of the 4 survey inputs is passed through its corresponding embedding layer.
        2. Concatenation: The numerical features and the survey embeddings are concatenated along the feature dimension.
        3. Projection Layer: The concatenated features are projected to the TCN channel dimension.
        4. TCN Layer: The projected features are passed through the Temporal Convolutional Network (TCN).
        5. SE Block: The output of the TCN is refined using a Squeeze-and-Excitation (SE) block.
        6. BiLSTM Layer: The refined features are processed by a Bidirectional LSTM to capture temporal dependencies.
        7. Attention Pooling: Attention pooling is applied to obtain a fixed-size dynamic feature vector.
        8. Static MLP: The static features are processed through a Static Multi-Layer Perceptron (MLP).
        9. Summary MLP: The summary features are processed through a Summary MLP.
        10. Gating Mechanism: A gating mechanism modulates the dynamic feature vector based on the static and summary features.
        11. Final Classification Head: The concatenated dynamic, static, and summary features are passed through
            the final classification head to produce logits.

        :param x_num: Tensor of shape (B, T, d_num) representing numerical time-series data.
        Where B is batch size, T is sequence length, and d_num is the number of numerical features.
        :param x_surv_list: List of 4 Tensors, each of shape (B, T), representing survey inputs.
        :param x_sta: Tensor of shape (B, d_sta) representing static features.
        Where B is batch size and d_sta is the number of static features.
        :param x_summ: Tensor of shape (B, d_summ) representing summary features.
        Where B is batch size and d_summ is the number of summary features.
        :return: Tensor of shape (B, num_classes) representing the logits for each class.
        """
        # 1. Embedding layer for surveys
        embs = [emb(surv) for emb, surv in zip(self.survey_embs, x_surv_list)]
        # 2. Concatenate numerical features and survey embeddings
        x = cat([x_num, *embs], dim=-1)

        # 3. Projection to TCN channels
        x = self.proj(x)        # ~> (B,T,C)
        # 4. TCN + 5. SE
        x = x.transpose(1, 2)   # ~> (B,C,T)
        x = self.tcn(x)
        x = self.se(x)
        x = x.transpose(1, 2)   # ~> (B,T,C)

        # 6. BiLSTM
        x, _ = self.lstm(x)     # ~> (B,T,2*hidden)

        # 7. Attention pooling
        dyn_vec = self.attn(x)  # ~> (B,2*hidden)
        # 8. Static features branch
        sta_vec = self.static_mlp(x_sta)    # ~> (B,64)
        # 9. Summary features branch
        summ_vec = self.summ_mlp(x_summ)    # ~> (B,64)

        # 9.1. Drop summary branch with probability p_drop_summ during training;
        #      If the model is in training mode and a random number is less than p_drop_summ,
        #      we set the summary vector to zero to effectively drop this branch for the current step;
        #      Why a random number? Simple, we want to introduce stochasticity in dropping the branch
        if self.training and rand(1, device=summ_vec.device) < self.p_drop_summ:
            summ_vec = zeros_like(summ_vec) # bye bye summary

        # 10. Gating mechanism to modulate dynamic features
        #     (we use a sigmoid gate here because we want to scale between 0 and 1)
        context = cat(tensors=[sta_vec, summ_vec], dim=-1)  # ~> (B,128)
        gate = sigmoid(self.gate_fc(context))               # ~> (B, 2H)
        dyn_vec *= gate                                     # modulate dynamic encoding

        # 11. Final classification head, concatenate all features
        z = cat([dyn_vec, sta_vec, summ_vec], dim=-1)
        # 12. Output logits, pass through head; shape ~> (B, num_classes)
        return self.head(z)

    def forward_with_surv_embs(self, x_num: Tensor, surv_embs: list[Tensor], x_sta: Tensor, x_summ: Tensor) -> Tensor:
        """
        Forward pass of the PainTCNBiLSTMAttn model with pre-computed survey embeddings.

        This method is similar to the standard forward pass but assumes that the survey embeddings
        have already been computed and are provided as input. This can be useful for scenarios
        where the survey embeddings are pre-processed or cached.
        :param x_num: Tensor of shape (B, T, d_num) representing numerical time-series data.
        :param surv_embs: List of 4 Tensors, each of shape (B, T, d_emb), representing pre-computed survey embeddings.
        :param x_sta: Tensor of shape (B, d_sta) representing static features.
        :param x_summ: Tensor of shape (B, d_summ) representing summary features.
        :return: Tensor of shape (B, num_classes) representing the logits for each class.
        """
        # surv_embs: list of 4 tensors, each (B, T, d_emb)
        x = cat([x_num, *surv_embs], dim=-1)     # (B,T,d_num+4*d_emb)
        x = self.proj(x)                         # (B,T,C)
        x = x.transpose(1, 2)                    # (B,C,T)
        x = self.tcn(x)
        x = self.se(x)
        x = x.transpose(1, 2)                    # (B,T,C)
        x, _ = self.lstm(x)                      # (B,T,2H)
        dyn_vec = self.attn(x)                   # (B,2H)
        sta_vec = self.static_mlp(x_sta)         # (B,64)
        summ_vec = self.summ_mlp(x_summ)         # (B,summ_out)

        if self.training and rand(1, device=summ_vec.device) < self.p_drop_summ:
            summ_vec = zeros_like(summ_vec)

        gate = sigmoid(self.gate_fc(cat([sta_vec, summ_vec], dim=-1)))  # (B,2H)
        dyn_vec = dyn_vec * gate
        z = cat([dyn_vec, sta_vec, summ_vec], dim=-1)
        return self.head(z)
