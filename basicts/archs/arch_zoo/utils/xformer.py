import torch


def data_transformation_4_xformer(history_data: torch.Tensor, future_data: torch.Tensor, start_token_len: int):
    """Transfer the data into the XFormer format.

    Args:
        history_data (torch.Tensor): history data with shape: [B, L1, N, C].
        future_data (torch.Tensor): future data with shape: [B, L2, N, C]. 
                                    L1 and L2 are input sequence length and output sequence length, respectively.
        start_token_length (int): length of the decoder start token. Ref: Informer paper.

    Returns:
        torch.Tensor: x_enc, input data of encoder (without the time features). Shape: [B, L1, N]
        torch.Tensor: x_mark_enc, time features input of encoder w.r.t. x_enc. Shape: [B, L1, C-1]
        torch.Tensor: x_dec, input data of decoder. Shape: [B, start_token_length + L2, N]
        torch.Tensor: x_mark_dec, time features input to decoder w.r.t. x_dec. Shape: [B, start_token_length + L2, C-1]
    """

    # get the x_enc
    x_enc = history_data[..., 0]            # B, L1, N
    # get the corresponding x_mark_enc
    # following previous works, we re-scale the time features from [0, 1) to to [-0.5, 0.5).
    x_mark_enc = history_data[:, :, 0, 1:] - 0.5    # B, L1, C-1

    # get the x_dec
    if start_token_len == 0:
        x_dec = torch.zeros_like(future_data[..., 0])     # B, L2, N
        # get the corresponding x_mark_dec
        x_mark_dec = future_data[..., :, 0, 1:] - 0.5                 # B, L2, C-1
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    else:
        x_dec_token = x_enc[:, -start_token_len:, :]            # B, start_token_length, N
        x_dec_zeros = torch.zeros_like(future_data[..., 0])     # B, L2, N
        x_dec = torch.cat([x_dec_token, x_dec_zeros], dim=1)    # B, (start_token_length+L2), N
        # get the corresponding x_mark_dec
        x_mark_dec_token = x_mark_enc[:, -start_token_len:, :]            # B, start_token_length, C-1
        x_mark_dec_future = future_data[..., :, 0, 1:] - 0.5          # B, L2, C-1
        x_mark_dec = torch.cat([x_mark_dec_token, x_mark_dec_future], dim=1)    # B, (start_token_length+L2), C-1

    return x_enc.float(), x_mark_enc.float(), x_dec.float(), x_mark_dec.float()
