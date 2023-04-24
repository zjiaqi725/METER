# METER
 ## Overview 
This repository contains the code for paper "METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection". The overall structure of METER is shown in the figure below. The main intuition is that the evolving stream data with different concepts should be identified and measured in for dynamic model evolution.
To achieve this goal, we propose four modules, respectively called Static Concept-aware Detector, Dynamic Shift-aware Detector, Intelligent Switching Controller and Offline Updating Strategy. 

![](../framework.pdf)

<img src="https://github.com/zjiaqi725/METER/blob/main/images/framework.pdf"  width="700">  

(a) Static Concept-aware Detector (SCD) is first trained on historical data to model the central concepts. (b) Intelligent Evolution Controller (IEC) timely measures the  concept uncertainty to determine the necessity of dynamic model evolution. (c) Dynamic Shift-aware Detector (DSD) dynamically updates SCD with the instance-aware parameter shift by considering the concept drift.  (d) Offline Updating Strategy (OUS) introduces an effective framework updating strategy according to the accumulated concept uncertainty given a sliding window.

 ## Implementation 
#### 1.Environment  
pytorch == 1.5.1  
torchvision == 0.6.1  
numpy == 1.21.5  
scipy == 1.4.1  
sklearn == 0.0

#### 2.Dataset  
We evaluate the proposed model on four publicly available datasets: (1)[DREAMER](https://zenodo.org/record/546113/accessrequest) (2)[The Stress Recognition in Automobile Drivers database (DRIVEDB)](https://www.physionet.org/content/drivedb/1.0.0/) (3)[Mahnob-HCI-tagging database (MAHNOB-HCI)](https://mahnob-db.eu/hci-tagging/) (4)[Wearable Stress and Affect Detection (WESAD)](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/). Detailed information about the datasets is summarized in the Table below.  

<img src="https://github.com/zjiaqi725/METER/blob/main/images/framework.pdf" width="700" >  

  #### 4.Train and Test the Model  
We write both training and evaluation process in the main.py, execute the following command to see the training and evaluation results.  
`python main.py`
