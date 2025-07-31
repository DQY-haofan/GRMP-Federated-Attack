# GRMP Attack for FedLLMs

The paper can be found here: [https://arxiv.org/abs/2507.01694](https://arxiv.org/abs/2507.01694)

## File Structure

```python
├── README.md # Project documentation
├── requirements.txt # Dependencies for the project
├── client.py # Client logic for user interaction
├── data_loader.py # Data loading and preprocessing
├── device_manager.py # Device detection and management
├── main.py # Main script for training and model execution
├── models.py # Deep learning model definitions
├── server.py # Server script for model deployment
└── visualizer.py # Visualization of data and results
```

## Dataset

The datasets can be downloaded in the following link.

https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

## Install Dependencies

```python
!pip install -r requirements.txt
```

## Run the Code

```python
# Import and run the main script
!python main.py
# Import and run the visualizer
!python visualizer.py
```

## Citation

```latex
@article{cai2025graph,
  title={Graph Representation-based Model Poisoning on Federated Large Language Models},
  author={Cai, Hanlin and Dong, Haofan and Wang, Houtianfu and Li, Kai and Akan, Ozgur B},
  journal={arXiv preprint arXiv:2507.01694},
  year={2025}
}
```
