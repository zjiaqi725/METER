# METER
 ## Overview 
This repository contains the code for paper "METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection". The overall structure of METER is shown in the figure below. The main intuition is that the evolving stream data with different concepts should be identified and measured in for dynamic model evolution.
To achieve this goal, we propose four modules, respectively called Static Concept-aware Detector, Dynamic Shift-aware Detector, Intelligent Switching Controller and Offline Updating Strategy. 

<img src="https://github.com/zjiaqi725/METER/blob/main/images/framework.png" width="900">  

(a) Static Concept-aware Detector (SCD) is first trained on historical data to model the central concepts. (b) Intelligent Evolution Controller (IEC) timely measures the  concept uncertainty to determine the necessity of dynamic model evolution. (c) Dynamic Shift-aware Detector (DSD) dynamically updates SCD with the instance-aware parameter shift by considering the concept drift.  (d) Offline Updating Strategy (OUS) introduces an effective framework updating strategy according to the accumulated concept uncertainty given a sliding window.

 ## Implementation 
#### 1.Environment  
pytorch == 1.5.1  
python == 3.7.6  
numpy == 1.21.5  
scipy == 1.4.1  
sklearn == 0.0  
pandas == 1.0.1  
hypnettorch == 0.0.4  
edl-pytorch == 0.0.2  

#### 2.Datasets description  
We select 14 real-world benchmark datasets from various domains that exhibit different types of concept drift, dimensions, number of data points, and anomaly rates. Four additional synthetic datasets were chosen to simulate different types and durations of concept drift according to the settings in [paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539348). The statistics of the datasets are summarised in the Table below.  

* Real-world datasets: (1) Anomaly detection datasets from the [UCI repository](https://archive.ics.uci.edu/ml/index.php) and [ODDS library]([https://zenodo.org/record/546113/accessrequest](http://odds.cs.stonybrook.edu/)), namely Ionosphere (Ion.), Pima, Satellite, Mammography (Mamm.). (2) Multi-aspect datasets of intrusion detection, namely [KDDCUP99 (KDD99)](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) and [NSL-KDD (NSL)](https://www.unb.ca/cic/datasets/nsl.html).
* Synthetic datasets



<img src="https://github.com/zjiaqi725/METER/blob/main/images/Statistical%20information%20of%20datasets.png" width="700" >  

  #### 4.Train and Test the Model  
We write both training and evaluation process in the main.py, execute the following command to see the training and evaluation results.  
`python main.py`
