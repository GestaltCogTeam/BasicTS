import torch

def date_normalize(data: torch.Tensor, time_of_day_size: int, day_of_week_size: int, day_of_month_size: int, day_of_year_size: int):
    """Normalize the date features.

    Args:
        data (torch.Tensor): the date features with shape: [B, L, C]
        time_of_day_size (int): the size of time in day
        day_of_week_size (int): the size of day in week
        day_of_month_size (int): the size of day in month
        day_of_year_size (int): the size of month in year
    """
    # time in day
    if time_of_day_size is not None and torch.all(data[:, :, 0].round() == data[:, :, 0]):   # not normalized
        data[:, :, 0] = data[:, :, 0] / (time_of_day_size - 1) - 0.5
    # day in week
    if day_of_week_size is not None and torch.all(data[:, :, 1].round() == data[:, :, 1]):
        data[:, :, 1] = data[:, :, 1] / (day_of_week_size-1) - 0.5
    # day in month
    if day_of_month_size is not None and torch.all(data[:, :, 2].round() == data[:, :, 2]):
        data[:, :, 2] = data[:, :, 2] / (day_of_month_size-1) - 0.5
    # month in year
    if day_of_year_size is not None and torch.all(data[:, :, 3].round() == data[:, :, 3]):
        data[:, :, 3] = data[:, :, 3] / (day_of_year_size-1) - 0.5

    return data

def data_transformation_4_xformer(history_data: torch.Tensor, future_data: torch.Tensor, start_token_len: int,
                                    time_of_day_size: int = None, day_of_week_size: int = None,
                                    day_of_month_size: int = None, day_of_year_size:int = None,
                                    embed_type: str= None):
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
    x_mark_enc = history_data[:, :, 0, 1:]    # B, L1, C-1
    if embed_type == 'timeF': # use as the time features
        x_mark_enc = date_normalize(x_mark_enc, time_of_day_size, day_of_week_size, day_of_month_size, day_of_year_size)

    # get the x_dec
    if start_token_len == 0:
        x_dec = torch.zeros_like(future_data[..., 0])     # B, L2, N
        x_mark_dec = future_data[..., :, 0, 1:]                 # B, L2, C-1
        x_mark_dec = date_normalize(x_mark_dec, time_of_day_size, day_of_week_size, day_of_month_size, day_of_year_size)
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    else:
        x_dec_token = x_enc[:, -start_token_len:, :]            # B, start_token_length, N
        x_dec_zeros = torch.zeros_like(future_data[..., 0])     # B, L2, N
        x_dec = torch.cat([x_dec_token, x_dec_zeros], dim=1)    # B, (start_token_length+L2), N
        # get the corresponding x_mark_dec
        x_mark_dec_token = x_mark_enc[:, -start_token_len:, :]            # B, start_token_length, C-1
        x_mark_dec_future = future_data[..., :, 0, 1:]          # B, L2, C-1
        x_mark_dec = torch.cat([x_mark_dec_token, x_mark_dec_future], dim=1)    # B, (start_token_length+L2), C-1
        x_mark_dec = date_normalize(x_mark_dec, time_of_day_size, day_of_week_size, day_of_month_size, day_of_year_size)

    return x_enc.float(), x_mark_enc.float(), x_dec.float(), x_mark_dec.float()
