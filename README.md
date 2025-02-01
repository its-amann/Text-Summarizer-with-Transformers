# Text Summarizer with Transformers

<div align="center">

*A Production-Ready Text Summarization Pipeline with Hugging Face Transformers*

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.78.0-green.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/Transformers-Latest-red.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Implementation Details](#-implementation-details)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Development](#-development)
- [License](#-license)

## 🔭 Overview

This project implements an end-to-end text summarization system using state-of-the-art transformer models. It features a complete MLOps pipeline from data ingestion to model deployment, with a FastAPI-based REST API for serving predictions.

## 🏗 Pipeline Architecture

The project follows a modular pipeline architecture with four main stages:

1. **Data Ingestion Pipeline**
   - Handles dataset downloading and preparation
   <div align="center">
     <img src="_asserts/data ingestion pipeline result.png" alt="Data Ingestion Results" width="800">
   </div>

2. **Data Transformation Pipeline**
   - Processes and transforms the data for training
   <div align="center">
     <img src="_asserts/showing the dataset_info dot json ,created in artifact folder by the pipeline.png" alt="Dataset Info" width="800">
   </div>

3. **Model Training Pipeline**
   - Loads pre-trained model
   <div align="center">
     <img src="_asserts/loading pretained model.png" alt="Loading Pre-trained Model" width="800">
   </div>
   
   - Fine-tunes on the prepared dataset
   <div align="center">
     <img src="_asserts/training the model.png" alt="Model Training" width="800">
   </div>
   
   - Tracks metrics and artifacts
   <div align="center">
     <img src="_asserts/metric result of the fine tunning.png" alt="Training Metrics" width="800">
   </div>

4. **Model Evaluation Pipeline**
   - Evaluates model performance
   <div align="center">
     <img src="_asserts/showing the metric artifacts made by the pipeline.png" alt="Evaluation Metrics" width="800">
   </div>

## 💫 FastAPI Integration

The model is served through a FastAPI application that provides:

- Interactive API documentation
<div align="center">
  <img src="_asserts/fastapi interface.png" alt="FastAPI Interface" width="800">
</div>

- Real-time inference endpoint
<div align="center">
  <img src="_asserts/infrencing using the fastapi , image shows the result we get in the fast api ui.png" alt="API Inference" width="800">
</div>

## 📁 Project Structure

```
text-summarize/
├── _asserts/               # Project screenshots and documentation assets
├── src/                    # Source code
│   └── textSummarizer/
│       ├── components/     # Pipeline components
│       ├── config/        # Configuration management
│       ├── constants/     # Project constants
│       ├── entity/        # Data models and entities
│       ├── logging/       # Logging configuration
│       ├── pipeline/      # Training and prediction pipelines
│       └── utils/         # Utility functions
├── research/              # Research notebooks and experiments
├── app.py                 # FastAPI application
├── main.py               # Training pipeline entry point
├── params.yaml           # Model and training parameters
├── setup.py             # Project installation setup
├── requirements.txt     # Project dependencies
├── Dockerfile          # Container configuration
└── template.py         # Template utilities
```

## ✨ Features

- **Complete MLOps Pipeline**: End-to-end pipeline from data ingestion to deployment
- **Production-Ready API**: FastAPI-based REST API with Swagger documentation
- **Modular Architecture**: Well-organized components for easy maintenance
- **Logging & Monitoring**: Comprehensive logging system with pipeline stage tracking
- **Parameter Management**: YAML-based parameter configuration
- **Docker Support**: Containerization for easy deployment

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/text-summarize.git
cd text-summarize
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Training Pipeline

Run the complete training pipeline:

```bash
python main.py
```

This will execute all pipeline stages:
1. Data Ingestion
2. Data Transformation
3. Model Training
4. Model Evaluation

### API Server

Start the FastAPI application:

```bash
python app.py
```

The API will be available at `http://localhost:8080` with interactive documentation at `/docs`

## 🔌 API Reference

### Endpoints

1. **GET /** 
   - Redirects to API documentation

2. **GET /train**
   - Triggers model training pipeline
   - Returns training status

3. **POST /predict**
   - Accepts text input for summarization
   - Returns generated summary

Example API request:
```python
import requests

response = requests.post(
    "http://localhost:8080/predict",
    json={"text": "Your long text here..."}
)
print(response.json())
```

## 🛠 Development

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers library
- FastAPI

### Environment Setup

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Configure parameters in `params.yaml`:
```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  weight_decay: 0.01
```

### Docker Support

Build and run the container:
```bash
docker build -t text-summarizer .
docker run -p 8080:8080 text-summarizer
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Made with ❤️ by Your Name</p>
  <p>
    <a href="https://github.com/yourusername">GitHub</a> •
    <a href="https://linkedin.com/in/yourusername">LinkedIn</a>
  </p>
</div> 