

# **DeepLog Anomaly Detection - HDFS, BGL, and OpenStack Datasets**  

This repository contains **three Jupyter notebooks** designed to examine **HDFS, BGL, and OpenStack datasets** using the **DeepLog anomaly detection model**. The implementation is based on the paper:  

> **DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning**  
> **Min Du, Feifei Li, Guineng Zheng, and Vivek Srikumar**  
> *2017 ACM SIGSAC Conference on Computer and Communications Security (CCS '17), 1285–1298.*  
> DOI: [10.1145/3133956.3134015](https://doi.org/10.1145/3133956.3134015)  

---

## **Repository Source and Modifications**  
This repository is **forked** from the original implementation at:  
[**DeepLog Repository**](https://github.com/nailo2c/deeplog)  

Several modifications were made to **accommodate new datasets and fix coding errors** in the original implementation.

---

## **Datasets**  
The **BGL dataset** used for testing can be found in the following Google Drive folder:  
[**BGL Dataset**](https://drive.google.com/drive/folders/1ASANK3UuLt7YPtM0xDK19QdOh9zrLyt_?usp=sharing)  

Other datasets (**HDFS and OpenStack**) are **already available** inside the `example/data/` directory in this repository.  

The **original datasets** were sourced from the **LogHub repository**:  
[**LogHub Datasets**](https://github.com/logpai/loghub)  

---

## **Setup Instructions**  
### **Testing with the BGL Dataset**  
1. **Create a directory in your Google Drive** with the name **"DeepLog > BGL"**.  
2. **Upload the BGL dataset** from the provided Google Drive folder into this **"DeepLog > BGL"** directory.  

### **Using Different Storage Mediums**  
If you store the dataset in a location other than Google Drive, **update the** `"input_dir"` **variable** in the **`bgl_preprocess.py`** file.  
This file can be found inside the `example/data/` directory.  

### **Running the Jupyter Notebooks**  
Each notebook corresponds to a different dataset:  
- **HDFS Notebook**: Runs DeepLog on the HDFS dataset.  
- **BGL Notebook**: Runs DeepLog on the BGL dataset.  
- **OpenStack Notebook**: Runs DeepLog on the OpenStack dataset.  

---

## **Implementation Details**  
DeepLog is a **deep learning-based anomaly detection model** that leverages **LSTM networks** to detect anomalies in system logs. The model learns normal log sequences during training and flags deviations as anomalies.  

### **Changes from the Original Implementation**  
- **Modified code blocks** to support **HDFS, BGL, and OpenStack datasets**.  
- **Fixed errors in the original implementation** to ensure smoother execution.  
- **Preprocessed datasets** for faster and more efficient anomaly detection.  

---
