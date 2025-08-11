# Spatiotemporal Topology-aware Transformer(SToT) - Pytorch Implementation
The repo is the official implementation for the paper: [How Will Arctic Shipping Emissions Evolve? Long-Term Forecasting with a Clustering-Driven Spatiotemporal Topology-aware Transformer](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5373225)

# Architecture
![Image](https://github.com/user-attachments/assets/b8dfbf23-8cba-42eb-930c-f7fed18b9fdb)

# Usage
**Installation**

Step1: Create a conda environment and activate it
```
conda create -n SToT python==3.8 --y
conda activate SToT
```
Step2: Install related version Pytorch following [here](https://pytorch.org/get-started/previous-versions/).
```
# Suggested
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
Step3: Install the required packages.
```
pip install -r requirements.txt
```

**Training & Test**
```
python main.py --pred_len 1800 --seq_len 360 --batch_size 32 --log_interval 15 
```

# Performance
![Image](https://github.com/user-attachments/assets/5a76a24b-6576-4a51-b955-4193e3621ef6)

# Hidden state visualization
![Image](https://github.com/user-attachments/assets/2e8ab030-e8c0-45ea-b647-d368a42b8f6b)
