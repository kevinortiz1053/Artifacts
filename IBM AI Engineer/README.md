# IBM AI Engineering — Course Projects

![IBM AI Engineering Professional Certificate](./ibm-ai-engineering-badge.png)

This repository collects the final projects I complete as part of IBM's **AI Engineering Professional Certificate**. Each project lives in its own notebook (and, where relevant, its own folder) and represents the capstone assignment for one course in the specialization.

More projects will be added here as I progress through the certificate.

## Projects

### 1. Rainfall Prediction Classifier — Australia Weather Data
**File:** [`FinalProject_AUSWeather.ipynb`](./FinalProject_AUSWeather.ipynb)

A supervised machine learning project that predicts whether it will rain using historical weather observations from the Australian Bureau of Meteorology (sourced via [Kaggle](https://www.kaggle.com)).

**What the project covers:**
- Data cleaning and handling missing values
- Avoiding data leakage by reframing the prediction target (predicting *today's* rain from data available *up to yesterday*)
- Feature engineering, including deriving a `Season` feature from the date
- Narrowing the dataset to a specific geographic region (Melbourne area) for a more localized model
- Building a preprocessing + modeling pipeline with `ColumnTransformer` (scaling numeric features, one-hot encoding categorical features)
- Hyperparameter tuning with `GridSearchCV` and stratified cross-validation
- Comparing two classifiers: **Random Forest** and **Logistic Regression**
- Evaluating models with classification reports, confusion matrices, and feature importance analysis

**Key libraries:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

**Result:** The tuned Random Forest model achieved roughly 84% test accuracy predicting daily rainfall for the Melbourne area.

---

### 2. Aircraft Damage Classification and Captioning — Pretrained Models
**File:** [`Final_Project_Classification_and_Captioning.ipynb`](./Final_Project_Classification_and_Captioning.ipynb)

A computer vision project combining transfer learning and vision-language models to automate aircraft damage inspection. The project has two parts: classifying damage type from images, and generating natural-language descriptions of that damage.

**What the project covers:**
- **Part 1 — Classification:** Fine-tuning a pretrained **VGG16** model (frozen convolutional base + custom dense classifier head) to classify aircraft images as either "dent" or "crack"
- Building `ImageDataGenerator` pipelines for training, validation, and test splits
- Training with the Adam optimizer and binary cross-entropy loss, then evaluating with test accuracy/loss, loss and accuracy curves, and visualized predictions
- **Part 2 — Captioning & Summarization:** Using Salesforce's **BLIP** (Bootstrapping Language-Image Pretraining) model to generate captions and summaries describing the damage in an image
- Wrapping BLIP in a custom Keras layer (`BlipCaptionSummaryLayer`) to integrate a Hugging Face Transformers model into a TensorFlow/Keras workflow

**Key libraries:** `tensorflow` / `keras`, `torch`, `transformers` (Hugging Face), `PIL`, `matplotlib`, `numpy`

**Result:** Produced a trained binary classifier for dent vs. crack detection, plus a pipeline that generates human-readable captions and summaries of aircraft damage images using BLIP.

---

## About This Repository

Each subfolder or notebook corresponds to the final/capstone project for one course in the IBM AI Engineering Professional Certificate. The certificate covers machine learning, deep learning, and AI engineering topics using Python, scikit-learn, and other common ML/DL frameworks.

| # | Project | Course Topic | Status |
|---|---------|--------------|--------|
| 1 | Rainfall Prediction Classifier | Machine Learning with Python | ✅ Complete |
| 2 | Aircraft Damage Classification and Captioning | Deep Learning / Computer Vision | ✅ Complete |

*(Table will be updated as new projects are added.)*
