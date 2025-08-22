import torch
import numpy as np
from torchinfo import summary


class HimNetRunner():
    def __init__(
        self,
        cfg: dict,
        device,
        scaler,
        log=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.scaler = scaler
        self.log = log

        self.clip_grad = cfg.get("clip_grad")

        self.batches_seen = 0

    def train_one_epoch(self, model, trainset_loader, optimizer, scheduler, criterion):
        model.train()
        batch_loss_list = []
        for x_batch, y_batch in trainset_loader:
            x = x_batch.to(self.device)
            y = y_batch.to(self.device)

            y_true = y[..., [0]]
            y_cov = y[..., 1:]

            output = model(x, y_cov, self.scaler.transform(y_true), self.batches_seen)
            y_pred = self.scaler.inverse_transform(output)

            self.batches_seen += 1

            loss = criterion(y_pred, y_true)
            batch_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            optimizer.step()

        epoch_loss = np.mean(batch_loss_list)
        scheduler.step()

        return epoch_loss

    @torch.no_grad()
    def eval_model(self, model, valset_loader, criterion):
        model.eval()
        batch_loss_list = []
        for x_batch, y_batch in valset_loader:
            x = x_batch.to(self.device)
            y = y_batch.to(self.device)

            y_true = y[..., [0]]
            y_cov = y[..., 1:]

            output = model(x, y_cov)
            y_pred = self.scaler.inverse_transform(output)

            loss = criterion(y_pred, y_true)
            batch_loss_list.append(loss.item())

        return np.mean(batch_loss_list)

    @torch.no_grad()
    def predict(self, model, loader):
        model.eval()
        y_list = []
        out_list = []

        for x_batch, y_batch in loader:
            x = x_batch.to(self.device)
            y = y_batch.to(self.device)

            y_true = y[..., [0]]
            y_cov = y[..., 1:]

            output = model(x, y_cov)
            y_pred = self.scaler.inverse_transform(output)

            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()
            out_list.append(y_pred)
            y_list.append(y_true)

        out = np.vstack(out_list).squeeze()  # (samples, out_steps, num_nodes)
        y = np.vstack(y_list).squeeze()

        return y, out

    def model_summary(self, model, dataloader):
        x_shape = next(iter(dataloader))[0].shape
        y_cov_shape = next(iter(dataloader))[1][..., 1:].shape

        return summary(
            model,
            [x_shape, y_cov_shape],
            verbose=0,  # avoid print twice
            device=self.device,
        )
