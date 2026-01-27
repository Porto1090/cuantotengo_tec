# CuantoTengo-Web-App

<p align="center">
  <a href="#research-context">Research Context</a> •
  <a href="#infrastructure">Infrastructure</a> •
  <a href="#installation">Installation</a>
</p>

<div style="text-align: center;">
  <img src="public/cuantotengo.gif" alt="CuantoTengo Demo" style="center">
</div>

CuantoTengo is a mobile-first inventory management tool that uses computer vision to automatically recognize and count products from a single image captured with a smartphone.
The system detects cans displayed on shelves, enabling small businesses to perform fast and accurate inventory counts without specialized hardware or barcode scanners.

## Research Context
Traditional inventory counting is time-consuming, error-prone, and often inaccessible for small retailers. CuantoTengo addresses this problem by leveraging advanced computer vision models to transform a simple photo into structured inventory data.

From a single image, the system:
- Detects individual products
- Classifies product types listing them in groups
- Counts instances per category
- Returns results in near real time

## Infrastructure

<table>
  <tr>
    <td><strong>Computer Vision</strong></td>
    <td>
      <img src="https://img.shields.io/badge/Flask-150458.svg?&logo=flask&logoColor=white" alt="Flask"/>
      <img src="https://img.shields.io/badge/Gradio-150458.svg?&logo=gradio&logoColor=white" alt="Gradio"/>
    </td>
  </tr>
  <tr>
    <td><strong>Backend | Data Analytics</strong></td>
    <td>
      <img src="https://img.shields.io/badge/NumPy-000000.svg?&logo=numpy&logoColor=white" alt="NumPy"/>
      <img src="https://img.shields.io/badge/Pandas-000000.svg?&logo=pandas&logoColor=white" alt="Pandas"/>
    </td>
  </tr>
  <tr>
    <td><strong>Computer Vision</strong></td>
    <td>
      <img src="https://img.shields.io/badge/OpenCV-150458.svg?&logo=opencv&logoColor=white" alt="OpenCV"/>
      <img src="https://img.shields.io/badge/PyTorch-150458.svg?&logo=pytorch&logoColor=white" alt="PyTorch"/>
      <img src="https://img.shields.io/badge/Ultralytics YOLO-150458.svg?&logo=pytorch&logoColor=white" alt="Ultralytics YOLO"/>
    </td>
  </tr>
</table>

## Project Structure

```bash
.
├── Call_Qualtrics_API.py
├── __pycache__
│   ├── azure_blob_storage.cpython-311.pyc
│   ├── keys.cpython-311.pyc
│   └── keys.cpython-313.pyc
├── app.py
├── azure_blob_storage.py
├── brand_model_3class.pt
├── classification_model copy.py
├── classification_model.py
├── data
│   ├── ...
├── inference
│   ├── __pycache__
│   ├── config.py
│   ├── detection
│   ├── final_aggregation.py
│   ├── image_utils.py
│   ├── main_inference.py
│   └── plots.py
├── keys.py
├── main.py
├── main_original.py
├── models
│   ├── bottle_can_cap_yolo_weights.pt
│   └── bottlefront_weights.pt
├── requirements.txt
├── services
│   ├── __pycache__
│   ├── blob_storage.py
│   ├── database.py
│   ├── setup_documentation.txt
│   └── whatsapp.py
└── venv
    └── ...
```

## Installation

### Prerequisites

- [Python3.11](https://www.python.org/downloads/release/python-3110/)

### Manual Installation

```bash
# Clone the repository
git clone https://github.mit.edu/LIFT-Lab/CuantoTengo-Web-App.git
```

### Set environment and run service

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (macOS / Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The Gradio app will be available at [http://localhost:7860/](http://localhost:7860/).

## Environment Variables Configuration 
Set these as your environment variables or create a `.env` file. The variables should be structured as follows:

```bash
AZURE_STORAGE_CONNECTION_STRING=...
AZURE_CONTAINER=...
```

See [`.env.example`][.env] for an example environment.

---

Developed by Eduardo Porto Morales [@Porto1090](https://github.com/Porto1090)

[.env]: /.env.example
