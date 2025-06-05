<<<<<<< HEAD
# penalty_prediction
=======
Penalty Kick Classifier ⚽

This web app classifies penalty kick directions into five categories using a Convolutional Neural Network with transfer learning (InceptionV3).

### 📂 Model Info
- Architecture: InceptionV3 (transfer learning)
- Trained on: Custom dataset of 5 directions
- Framework: TensorFlow + Streamlit


## Model Performance

After training the InceptionV3 model with fine-tuning, the following validation metrics were achieved:

| Metric             | Value  |
|--------------------|--------|
| Validation Accuracy| 0.9179 |
| Validation Precision| 0.9179 |
| Validation Recall  | 0.9104 |
| Validation AUC     | 0.9946 |
| Validation F1 Score| 0.9141 |


### 🚀 How to Run Locally
```
cd app
streamlit run app.py
```

### 📁 Project Structure
- `src/`: preprocessing, model building modules
- `train.py`: end-to-end training script
- `models/`: saved `.h5` model file
- `app/`: Streamlit deployment code
>>>>>>> 4a0d9db (added folders required)
