# Text Emotion Classification App

This is a machine learning application that classifies text into different emotions using a Random Forest Classifier. The application includes both the training model and a Streamlit-based web interface for easy interaction.

## Project Structure

```
.
├── model.ipynb          # Jupyter notebook containing the model training code
├── app.py              # Streamlit web application
├── train.txt          # Training dataset
├── trained_model.pkl   # Saved trained model
├── vectorizer.pkl     # Saved text vectorizer
└── README.md          # This file
```

## Features

- Text emotion classification using Random Forest Classifier
- Interactive web interface built with Streamlit
- Real-time predictions with confidence scores
- Pre-trained model for immediate use

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

2. Install required packages:
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to:
- Local: http://localhost:8501
- Network: http://[your-ip]:8501

3. Enter text in the input area and click "Classify Text" to get predictions.

## Model Training

The model is trained using:
- CountVectorizer for text feature extraction
- Random Forest Classifier for classification
- Train-test split of 80-20

To retrain the model, run all cells in `model.ipynb`.

## Files Description

- `model.ipynb`: Contains the complete model training pipeline
- `app.py`: Streamlit web application code
- `train.txt`: Dataset containing text samples and their emotion labels
- `trained_model.pkl`: Serialized trained Random Forest model
- `vectorizer.pkl`: Serialized CountVectorizer for text preprocessing

## Requirements

- Python 3.6+
- streamlit
- pandas
- numpy
- scikit-learn
- joblib

## License

This project is open source and available under the MIT License.
