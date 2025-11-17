# Deploying a Scalable ML Pipeline with FastAPI

**GitHub Repository**: https://github.com/bpolich-code/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

## Project Overview
This project implements a complete machine learning pipeline that:
- Trains a Random Forest classifier on Census Bureau data
- Creates a RESTful API using FastAPI
- Implements CI/CD with GitHub Actions
- Includes comprehensive unit tests
- Analyzes model performance across data slices

## Model Performance
- **Precision**: 0.7915
- **Recall**: 0.6136
- **F1 Score**: 0.6913

## Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python train_model.py`
4. Run tests: `pytest test_ml.py -v`
5. Start API: `uvicorn main:app --reload`

## API Usage
- GET `/`: Welcome message
- POST `/predict`: Income prediction endpoint

## Files
- `ml/model.py`: ML model functions
- `ml/data.py`: Data processing functions  
- `train_model.py`: Training pipeline
- `main.py`: FastAPI application
- `test_ml.py`: Unit tests
- `model_card.md`: Model documentation
