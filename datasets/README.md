# Datasets

## Datasets Downloading

The raw data can be downloaded at this [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp) or [Baidu Yun](https://pan.baidu.com/s/18qonT9l1_NbvyAgpD4381g)(password: 0lrk), and should be unzipped to datasets/raw_data/.

## Datasets Description

### ETT (Electricity Transformer Temperature) Datasets

**Source**: [Informer: Beyond Efﬁcient Transformer for Long Sequence Time-Series Forecasting, AAAI 2021](https://github.com/zhouhaoyi/Informer2020). [Data Link](https://github.com/zhouhaoyi/ETDataset)

**Description**: ETT is a series of datasets, including ETTh1, ETTm1, ETTh2, and ETTm2, where "1" and "2" represent different transformers. "h" and "m" represent different sampling frequency (every 15 minute and every hour).

**Period**: 2016/7/1 0:00 -> 2018/6/26 19:45

**Number of Time Steps**: 69680(ETTm1, ETTm2), 17420(ETTh1, ETTh2)

**Variates**: HUFL (High UseFul Load), HULL (High UseLess Load), MUFL (Middle UseFul Load), MULL (Middle UseLess Load), LUFL (Low UseFul Load), LULL (Low UseLess Load), OT (Oil Temperature). 
The first six variates are overload data, while the last one is oil temperature.

**Typical Settings**: 

- **M**: Multivariate time series forecasting. Input all 7 features, predict all 7 features.
- **S**: Univariate time series forecasting. Input 1 feature (oil temperature default), predict the corresponding feature.
- **MS**: Multivariate predict univariate. Input all 7 features, predict 1 feature (oil temperature default).
