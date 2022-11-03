# Datasets

## Datasets Downloading

The raw data can be downloaded at this [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp) or [Baidu Yun](https://pan.baidu.com/s/18qonT9l1_NbvyAgpD4381g)(password: 0lrk), and should be unzipped to datasets/raw_data/.

## Datasets Description

### 1. ETT (Electricity Transformer Temperature) Datasets

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

### 2. Electricity

**Source**: [Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks, SIGIR'18](https://github.com/laiguokun/LSTNet). [Data Link](https://github.com/laiguokun/multivariate-time-series-data).

**Description**: Electricity contains the electricity consumption in kWh from 2012 to 2014. The data is converted to reflect hourly consumption. It contains 321 clients. 

**Period**: 2012 -> 2014

**Number of Time Steps**:  26304

**Variates**: Each variate represents the electricity consumption of a client.

**Typical Settings**: 

- **M**: Multivariate time series forecasting. Input all 321 features, predict all 321 features.

### 3. Weather

**Source**: [Informer: Beyond Efﬁcient Transformer for Long Sequence Time-Series Forecasting, AAAI 2021](https://github.com/zhouhaoyi/Informer2020). [Data Link](https://github.com/zhouhaoyi/Informer2020)

**Description**: Weather contains  4 years‘ local climatological data from 2010 to 2013, where data points are collected every 1 hour.
Kindly note that there are multiple versions of Weather dataset in different papers, such as AutoFormer and Informer.
We choose the version of Informer.

**Period**: 2010/1/1 0:00 -> 12/31/2013 23:00

**Number of Time Steps**: 35064

**Variates**: Visibility, DryBulbFarenheit, DryBulbCelsius, WetBulbFarenheit, DewPointFarenheit, DewPointCelsius, RelativeHumidity, Windspeed, WindDirection, StationPressure, Altimeter and WetBulbCelsius.

**Typical Settings**: 

- **M**: Multivariate time series forecasting. Input all 12 features, predict all 12 features.
- **S**: Univariate time series forecasting. Input 1 feature (wet bulb default), predict the corresponding feature.
- **MS**: Multivariate predict univariate. Input all 12 features, predict 1 feature (wet bulb default).

### 4. Solar-Energy

**Source**: [Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks, SIGIR'18](https://github.com/laiguokun/LSTNet). [Data Link](https://github.com/laiguokun/multivariate-time-series-data). 

**Description**:  The solar power production records in the year of 2006, which is sampled every 10 minutes from 137 PV plants in Alabama State.

**Period**: 2006/01/01 00:00 -> 2006/12/31 23:50

**Number of Time Steps**: 52560

**Variates**: Each variate represents the power output of a PV plant.

**Typical Settings**:

- **M**: Multivariate time series forecasting. Input all 137 features, predict all 137 features.

### 5. Exchange-Rate

**Source**: [Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks, SIGIR'18](https://github.com/laiguokun/LSTNet). [Data Link](https://github.com/laiguokun/multivariate-time-series-data). 

**Description**:  The collection of the daily exchange rates of eight countries including Australia, British, Canada, Switzerland, China, Japan, New Zealand, and Singapore.

**Period**: 1990 -> 2016

**Number of Time Steps**: 7588

**Variates**: Exchange rates of Australia, British, Canada, Switzerland, China, Japan, New Zealand, and Singapore.

**Typical Settings**:

- Multivariate time series forecasting. Input all 8 features, predict all 8 features.

### 6. Beijing Air Quality

**Source**: https://quotsoft.net/air/

**Description**:  The air quality index of Beijing, including seven indicators: AQI(Air quality index), PM2.5, PM10, SO2, NO2, O3, and CO. The sampling rate is 6 hours.

**Period**: 2014 -> 2018

**Number of Time Steps**: 6000

**Variates**:AQI, PM2.5, PM10, SO2, NO2, O3, and CO.

**Typical Settings**:

- Multivariate time series forecasting. Input all 7 features, predict all 7 features.
