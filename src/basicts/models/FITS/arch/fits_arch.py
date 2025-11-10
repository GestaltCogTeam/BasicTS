import torch
import torch.nn as nn


class FITS(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.individual = config.individual
        self.channels = config.enc_in

        self.cut_freq=config.cut_freq
        self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.cut_freq, int(self.cut_freq*self.length_ratio)).to(torch.cfloat))
        else:
            self.freq_upsampler = nn.Linear(
                self.cut_freq, int(self.cut_freq * self.length_ratio)).to(torch.cfloat) # complex layer for frequency upcampling]

    def forward(self, inputs: torch.Tensor):

        inputs = self.revin(inputs, "norm")

        low_specx = torch.fft.rfft(inputs, dim=1)
        low_specx[:,self.cut_freq:]=0 # LPF
        low_specx = low_specx[:,0:self.cut_freq,:] # LPF
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0),int(self.cut_freq*self.length_ratio),low_specx.size(2)],dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:,:,i]=self.freq_upsampler[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)

        low_specxy = torch.zeros([low_specxy_.size(0),int((self.seq_len+self.pred_len)/2+1),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_ # zero padding
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        low_xy=low_xy * self.length_ratio # energy compemsation for the length change
        
        xy = self.revin(low_xy, "denorm")
        return xy
