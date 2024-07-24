# METER
This repository contains the code for paper "METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection", accepted by VLDB 2024.   
 ## Overview 
The overall structure of METER is shown in the figure below. The main intuition is that the evolving stream data with different concepts should be identified and measured in for dynamic model evolution.
To achieve this goal, we propose four modules, respectively called Static Concept-aware Detector, Dynamic Shift-aware Detector, Intelligent Switching Controller and Offline Updating Strategy. 

<img src="https://github.com/zjiaqi725/METER/blob/main/images/framework.png" width="900">  

(a) Static Concept-aware Detector (SCD) is first trained on historical data to model the central concepts. (b) Intelligent Evolution Controller (IEC) timely measures the  concept uncertainty to determine the necessity of dynamic model evolution. (c) Dynamic Shift-aware Detector (DSD) dynamically updates SCD with the instance-aware parameter shift by considering the concept drift.  (d) Offline Updating Strategy (OUS) introduces an effective framework updating strategy according to the accumulated concept uncertainty given a sliding window.

 ## Implementation 
#### 1.Environment  
pytorch == 1.5.1  
python == 3.7.6  
hypnettorch == 0.0.4  
edl-pytorch == 0.0.2  

#### 2.Datasets Description  
We select 14 real-world benchmark datasets from various domains that exhibit different types of concept drift, dimensions, number of data points, and anomaly rates. Four additional synthetic datasets were chosen to simulate different types and durations of concept drift according to the settings in [paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539348). The statistics of the datasets are summarised in the Table below.  

* Real-world datasets: (1) Anomaly detection datasets from the [UCI repository](https://archive.ics.uci.edu/ml/index.php) and [ODDS library](http://odds.cs.stonybrook.edu/), namely Ionosphere (Ion.), Pima, Satellite, Mammography (Mamm.). (2) A large public dataset [BGL](https://www.usenix.org/cfdr-data) dataset, consisting of log messages collected from a BlueGene/L supercomputer system at Lawrence Livermore National Labs. To facilitate analysis, each log message is processed into the structured data format. (3) Multi-aspect datasets of intrusion detection, namely [KDDCUP99 (KDD99)](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) and [NSL-KDD (NSL)](https://www.unb.ca/cic/datasets/nsl.html). (4) Time-series datasets, namely
NYC taxicab (NYC), CPU utilization (CPU), Machine temperature (M.T.) and Ambient temperature (A.T.) from the [Numenta anomaly detection benchmark (NAB)](https://github.com/numenta/NAB). (5) Real-world streaming datasets [INSECTS](https://sites.google.com/view/uspdsrepository).
* Synthetic datasets: Four [synthetic datasets](https://www.dropbox.com/sh/a3fhtp9zjjujrwa/AAD_4wkFaULuK-uJinbtw81Oa?dl=0) are created for simulating complex anomaly detection scenarios or data streams. It randomly sets categories as anomaly targets to simulate concepts and sets the duration of each concept randomly to simulate two types of concept drift: "abrupt and recurrent" and "gradual and recurrent".

<img src="https://github.com/zjiaqi725/METER/blob/main/images/Dataset%20statistics.png" width="600" >  

#### 3.Code Running  
We write both training and evaluation process in the main.py, execute the following command to see the training and evaluation results.  
`python main.py --dataset ionosphere --epochs 1000 --train_rate 0.2 --mode hybrid+edl --thres_rate 0.1`

#### 4.Command line options
* mode: type of model, one of ["static", "dynamic", "hybrid", "hybrid+edl"]
* emb_rate: rate of embedding dim to input dim, default=2
* train_rate: rate of training set, default=0.1
* epochs: number of epochs, default=1000
* thres_rate: threshold rate for the pseudo labels from SCD, default=0.05
* uncertainty_threshold: threshold of the concept uncertainty, default=0.1
* uncertainty_avg_threshold: threshold of the offline updating strategy, default=0.1

## Citation
Read our [paper](https://www.vldb.org/pvldb/vol17/p794-zhu.pdf) for more information. If you use our method, please cite us using
```bash
@article{zhu2023meter,
  title={METER: A Dynamic Concept Adaptation Framework for Online Anomaly Detection},
  author={Zhu, Jiaqi and Cai, Shaofeng and Deng, Fang and Ooi, Beng Chin and Zhang, Wenqiao},
  journal={Proceedings of the VLDB Endowment},
  volume={17},
  number={4},
  pages={794--807},
  year={2023},
  publisher={VLDB Endowment}
}
```

## Acknowledgments
This project is based on the following open-source projects. We thank their authors for making the source code publicly available.  
* [Hypernetworks](https://github.com/chrhenning/hypnettorch)
* [Evidential Deep Learning](https://github.com/teddykoker/evidential-learning-pytorch)
