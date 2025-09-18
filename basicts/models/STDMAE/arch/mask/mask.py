import torch
from torch import nn

from .patch import PatchEmbedding
from .maskgenerator import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers


def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class Mask(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout,  mask_ratio, encoder_depth, decoder_depth,spatial=False, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.spatial=spatial
        self.selected_feature = 0

        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat=None
        # encoder specifics
        # # patchify & embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding()

        # encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        # decoder specifics
        # transform layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # # decoder
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        # import here to fix bugs related to set visible device
        from timm.models.vision_transformer import trunc_normal_
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, mask=True):
        """

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, C, P * L],
                                                which is used in the Pre-training.
                                                P is the number of patches.
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        # patchify and embed input
        if mask:
            if self.spatial:
                patches = self.patch_embedding(long_term_history)  # B, N, d, P
                patches = patches.transpose(-1, -2)  # B, N, P, d
                batch_size, num_nodes, num_time,num_dim  =  patches.shape
                patches,self.pos_mat = self.positional_encoding(patches)        # mask
                Maskg=MaskGenerator(patches.shape[1], self.mask_ratio)
                unmasked_token_index, masked_token_index = Maskg.uniform_rand()
                encoder_input = patches[:, unmasked_token_index, :, :]
                encoder_input=encoder_input.transpose(-2,-3)
                #print(encoder_input.shape)
                hidden_states_unmasked = self.encoder(encoder_input)
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size,num_time, -1, self.embed_dim)

            if not self.spatial:
                patches = self.patch_embedding(long_term_history)  # B, N, d, P
                patches = patches.transpose(-1, -2)  # B, N, P, d
                batch_size, num_nodes, num_time,num_dim  =  patches.shape

                # positional embedding
                patches,self.pos_mat = self.positional_encoding(patches)        # mask
                Maskg=MaskGenerator(patches.shape[2], self.mask_ratio)
                unmasked_token_index, masked_token_index = Maskg.uniform_rand()
                encoder_input = patches[:, :, unmasked_token_index, :]
                #print(encoder_input.shape)
                hidden_states_unmasked = self.encoder(encoder_input)
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)

        else:
            batch_size, num_nodes, _, _ = long_term_history.shape
            # patchify and embed input
            patches = self.patch_embedding(long_term_history)     # B, N, d, P
            patches = patches.transpose(-1, -2)         # B, N, P, d
            # positional embedding
            patches,self.pos_mat = self.positional_encoding(patches)# B, N, P, d
            #print(self.pos_mat.shape)
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches# B, N, P, d
            if self.spatial:
                encoder_input=encoder_input.transpose(-2,-3)# B,  P,N, d
            hidden_states_unmasked = self.encoder(encoder_input)# B,  P,N, d/# B, N, P, d
            if self.spatial:
                hidden_states_unmasked=hidden_states_unmasked.transpose(-2,-3)# B, N, P, d
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)# B, N, P, d
            return hidden_states_unmasked, unmasked_token_index, masked_token_index
        # encoding

        return hidden_states_unmasked,  unmasked_token_index, masked_token_index

    def decoding(self, hidden_states_unmasked, masked_token_index):

        # encoder 2 decoder layer
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)# B, N, P, d/# B,P, N,  d
        # B,N*r,P,d
        if self.spatial:
            # TO WORK SPATIAL:
            batch_size,  num_time,num_nodes, _ = hidden_states_unmasked.shape# B,P, N,  d
            unmasked_token_index=[i for i in range(0,len(masked_token_index)+num_nodes) if i not in masked_token_index ]
            hidden_states_masked = self.pos_mat[:,masked_token_index,:,:]# B, N*r,P,  d
            hidden_states_masked=hidden_states_masked.transpose(-2,-3)# B, P, N*r, d

            hidden_states_masked+=self.mask_token.expand(batch_size, num_time, len(masked_token_index), hidden_states_unmasked.shape[-1])# B, P, N*r, d
            hidden_states_unmasked+=self.pos_mat[:,unmasked_token_index,:,:].transpose(-2,-3)# B,  P,N*(1-r), d
            hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, P, N, d

            # decoding
            hidden_states_full = self.decoder(hidden_states_full)# B, P, N, d
            hidden_states_full = self.decoder_norm(hidden_states_full)# B, P, N, d
            # prediction (reconstruction)
            reconstruction_full = self.output_layer(hidden_states_full.view(batch_size,num_time, -1,  self.embed_dim))# B, P, N, L
        else:
            batch_size, num_nodes, num_time, _ = hidden_states_unmasked.shape
            unmasked_token_index=[i for i in range(0,len(masked_token_index)+num_time) if i not in masked_token_index ]
            hidden_states_masked = self.pos_mat[:,:,masked_token_index,:]
            hidden_states_masked+=self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1])
            hidden_states_unmasked+=self.pos_mat[:,:,unmasked_token_index,:]
            hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d

            # decoding
            hidden_states_full = self.decoder(hidden_states_full)
            hidden_states_full = self.decoder_norm(hidden_states_full)

            # prediction (reconstruction)
            reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))

        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index,
                                        masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # get reconstructed masked tokens
        if self.spatial:
            batch_size,  num_time,num_nodes, _ = reconstruction_full.shape# B, P, N, L
            reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     # B, P, r*N, L
            reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_time, -1)     # B, P, r*N*L
    
            label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
            label_masked_tokens = label_full[:, masked_token_index, :, :].transpose(1, 2).contiguous() # B,  P,N*r, L
            label_masked_tokens = label_masked_tokens.view(batch_size,  num_time,-1)  # B, P, r*N*L
    
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            batch_size, num_nodes, num_time, _ = reconstruction_full.shape

            reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     # B, N, r*P, d
            reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)     # B, r*P*d, N

            label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
            label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, d
            label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*d, N

            return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        
        # feed forward
        if self.mode == "pre-train":
            # encoding
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(history_data)
            # decoding
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index)
            # for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)

            return reconstruction_masked_tokens.unsqueeze(-1), label_masked_tokens.unsqueeze(-1)
        else:
            hidden_states_full, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full
