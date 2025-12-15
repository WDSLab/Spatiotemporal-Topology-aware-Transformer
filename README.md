# Spatiotemporal Topology-aware Transformer (SToT)  <sub>(Transportation Research Part D: Transport and Environment)</sub>
The repo is the official implementation for the paper: [How Will Arctic Shipping Emissions Evolve? Long-Term Forecasting with a Clustering-Driven Spatiotemporal Topology-aware Transformer](https://doi.org/10.1016/j.trd.2025.105134)

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
<img width="1503" height="1724" alt="Image" src="https://github.com/user-attachments/assets/22e307d2-6946-453f-8e23-18bdf93a76ca" />

# Future estimation
<img width="975" height="765" alt="Image" src="https://github.com/user-attachments/assets/0bda1fbd-fa41-44b8-941d-c07fedc54200" />

# Hidden state visualization
<img width="5701" height="5933" alt="Image" src="https://github.com/user-attachments/assets/dfaa5553-f349-4d0f-80b6-a53c0a460d4d" />
