from torch import nn
from basicts.losses import masked_mae


def megacrn_loss(prediction, target, query, pos, neg, null_val):
    separate_loss = nn.TripletMarginLoss(margin=1.0)
    compact_loss = nn.MSELoss()
    criterion = masked_mae

    loss1 = criterion(prediction, target, null_val)
    loss2 = separate_loss(query, pos.detach(), neg.detach())
    loss3 = compact_loss(query, pos.detach())
    loss = loss1 + 0.01 * loss2 + 0.01 * loss3

    return loss
