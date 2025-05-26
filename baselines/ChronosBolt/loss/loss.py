def fake_loss(loss):
    # NOTE: Chronos integrates the cross-entropy loss calculation in the forward method
    return loss