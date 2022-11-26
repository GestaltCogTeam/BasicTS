# Datasets

## Datasets Downloading

The raw data can be downloaded at this [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp) or [Baidu Yun](https://pan.baidu.com/s/11d_am76_orMTV2vNejmuyg?pwd=v3ii)(password: v3ii), and should be unzipped to datasets/raw_data/.

## Datasets Description

### 1. ETT (Electricity Transformer Temperature) Datasets

**Source**: [Informer: Beyond Efﬁcient Transformer for Long Sequence Time-Series Forecasting, AAAI 2021](https://github.com/zhouhaoyi/Informer2020). [Data Link](https://github.com/zhouhaoyi/ETDataset)

**Description**: ETT is a series of datasets, including ETTh1, ETTm1, ETTh2, and ETTm2, where "1" and "2" represent different transformers. "h" and "m" represent different sampling frequency (every 15 minute and every hour).

**Period**: 2016/7/1 0:00 -> 2018/6/26 19:45. Many of the past works only use the first 20 months of data, e.g., Informer and Autoformer. In BasicTS, we use the full set of data.

**Dataset Split**: 6:2:2.

**Number of Time Steps**: 69680(ETTm1, ETTm2), 17420(ETTh1, ETTh2).

**Variates**: HUFL (High UseFul Load), HULL (High UseLess Load), MUFL (Middle UseFul Load), MULL (Middle UseLess Load), LUFL (Low UseFul Load), LULL (Low UseLess Load), OT (Oil Temperature). 
The first six variates are overload data, while the last one is oil temperature.

**Typical Settings**: 

- **M**: Multivariate time series forecasting. Input all 7 features, predict all 7 features. BasicTS use this setting as default for ETT datasets.
- **S**: Univariate time series forecasting. Input 1 feature (oil temperature default), predict the corresponding feature.
- **MS**: Multivariate predict univariate. Input all 7 features, predict 1 feature (oil temperature default).

### 2. Electricity

**Source**: [Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks, SIGIR'18](https://github.com/laiguokun/LSTNet). [Data Link](https://github.com/laiguokun/multivariate-time-series-data).

**Description**: Electricity records the electricity consumption in kWh every 1 hour from 2012 to 2014. It contains 321 clients. 

**Period**: 2012 -> 2014

**Dataset Split**: 7:1:2.

**Number of Time Steps**:  26304

**Variates**: Each variate represents the electricity consumption of a client.

**Typical Settings**: 

- **M**: Multivariate time series forecasting. Input all 321 features, predict all 321 features.

### 3. Weather

**Source**: [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting, NeurIPS 2021](https://github.com/thuml/Autoformer). [Data Link](https://github.com/thuml/Autoformer)

**Description**: Weather is recorded every 10 minutes for 2020 whole year, which contains 21 meteorological indicators, such as air temperature, humidity, etc. Kindly note that there are multiple versions of Weather dataset in different papers, such as AutoFormer and Informer.
We choose the version of AutoFormer.

**Period**: 2020/1/1 0:10 -> 2021/1/1 0:00

**Dataset Split**: 7:1:2

**Number of Time Steps**: 52696

**Variates**: 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)','H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR', 'PAR', 'max. PAR', 'Tlog (degC)', 'OT']

Visibility, DryBulbFarenheit, DryBulbCelsius, WetBulbFarenheit, DewPointFarenheit, DewPointCelsius, RelativeHumidity, Windspeed, WindDirection, StationPressure, Altimeter and WetBulbCelsius.

**Typical Settings**:

- **M**: Multivariate time series forecasting. Input all 21 features, predict all 21 features.
- **S**: Univariate time series forecasting. Input 1 feature (wet bulb default), predict the corresponding feature.
- **MS**: Multivariate predict univariate. Input all 21 features, predict 1 feature (wet bulb default).

### 4. Exchange Rate

**Source**: [Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks, SIGIR'18](https://github.com/laiguokun/LSTNet). [Data Link](https://github.com/laiguokun/multivariate-time-series-data). 

**Description**:  The collection of the daily exchange rates of eight countries including Australia, British, Canada, Switzerland, China, Japan, New Zealand, and Singapore.

**Period**: 1990 -> 2016

**Dataset Split**: 7:1:2

**Number of Time Steps**: 7588

**Variates**: Exchange rates of Australia, British, Canada, Switzerland, China, Japan, New Zealand, and Singapore.

**Typical Settings**:

- Multivariate time series forecasting. Input all 8 features, predict all 8 features.

### 5. Beijing Air Quality

**Source**: https://quotsoft.net/air/

**Description**:  The air quality index of Beijing, including seven indicators: AQI(Air quality index), PM2.5, PM10, SO2, NO2, O3, and CO. The sampling rate is 1 hours. Missing values are filled using linear interpolation.

**Period**: 2014 -> 2018

**Dataset Split**: 6:2:2

**Number of Time Steps**: 36000

**Variates**: AQI, PM2.5, PM10, SO2, NO2, O3, and CO.

**Typical Settings**:

- Multivariate time series forecasting. Input all 7 features, predict all 7 features.

### 6. METR-LA

**Source**: [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting, ICLR'18](https://github.com/liyaguang/DCRNN). [Data Link](https://github.com/liyaguang/DCRNN).

**Description**: METR-LA is a traffic speed dataset collected from loop-detectors located on the LA County road network. It contains data of 207 sensors over a period of 4 months from Mar 2012 to Jun 2012. The traffic information is recorded at the rate of every 5 minutes. METR-LA also includes a sensor graph to indicate dependencies between sensors. DCRNN computes the pairwise road network distances between sensors and build the adjacency matrix using a thresholded Gaussian kernel. Details can be found in the [paper](https://arxiv.org/pdf/1707.01926.pdf).

**Period**: 2012/3/1 0:00:00 -> 2012/6/27 23:55:00

**Number of Time Steps**: 34272

**Dataset Split**: 7:1:2.

**Variates**: Each variate represents the traffic speed of a sensor.

**Typical Settings**:

- Multivariate time series forecasting. Input all 207 features, predict all 207 features.

### 7. PEMS-BAY

**Source**: [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting, ICLR'18](https://github.com/liyaguang/DCRNN). [Data Link](https://github.com/liyaguang/DCRNN).

**Description**: PEMS-BAY is a traffic speed dataset collected from California Transportation Agencies (CalTrans) Performance Measurement System (PeMS). It contains data of 325 sensors in the Bay Area over a period of 6 months from Jan 2017 to June 2017. The traffic information is recorded at the rate of every 5 minutes. PEMS-BAY also includes a sensor graph to indicate dependencies between sensors. DCRNN computes the pairwise road network distances between sensors and build the adjacency matrix using a thresholded Gaussian kernel. Details can be found in the [paper](https://arxiv.org/pdf/1707.01926.pdf).

**Period**: 2017/1/1 0:00:00 -> 2017/6/30 23:55:00

**Number of Time Steps**: 52116

**Dataset Split**: 7:1:2.

**Variates**: Each variate represents the traffic speed of a sensor.

**Typical Settings**:

- Multivariate time series forecasting. Input all 325 features, predict all 325 features.

### 8. PEMS0X

**Source**: [Learning dynamics and heterogeneity of spatial-temporal graph data for traffic forecasting, TKDE'21](https://ieeexplore.ieee.org/abstract/document/9346058). [Data Link](https://github.com/guoshnBJTU/ASTGNN/tree/main/data).

**Description**: PEMS0X is a series of traffic flow dataset, including PEMS03, PEMS04, PEMS07, and PEMS08. X represents the code of district where the data is collected. The traffic information is recorded at the rate of every 5 minutes. Similar to METR-LA and PEMS-BAY, PEMS0X also includes a sensor graph to indicate dependencies between sensors. The details of the computation of the adjacency matrix can be found in the [ASTGCN](https://ojs.aaai.org/index.php/AAAI/article/view/3881/3759).


**Period**:

- PEMS03: 2018/09/01 -> 2018/11/30
- PEMS04: 2018/01/01 -> 2018/2/28
- PEMS07: 2017/05/01 -> 2017/08/31
- PEMS08: 2016/07/01 -> 2016/08/31

**Number of Time Steps**:

- PEMS03: 26208
- PEMS04: 16992
- PEMS07: 28224
- PEMS08: 17856

**Dataset Split**: 6:2:2.

**Variates**: Each variate represents the traffic flow of a sensor.

**Number of Variates**:

- PEMS03: 358
- PEMS04: 307
- PEMS07: 883
- PEMS08: 170

**Typical Settings**:

- Multivariate time series forecasting. Input all features, predict all features.
