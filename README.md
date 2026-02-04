# Heart Disease Risk Prediction - Logistic Regression Implementation

## Project Overview

This project implements a **logistic regression model from scratch** to predict heart disease risk. The implementation includes manual gradient descent optimization, comprehensive performance metrics, and decision boundary visualizations for binary classification.

## Dataset

- **Source**: Heart Disease Prediction Dataset
- **Total Samples**: 270 patients
- **Features**: 14 attributes including Age, Cholesterol, Blood Pressure, Max Heart Rate, ST Depression, and Number of Vessels (fluoroscopy)
- **Target Variable**: Heart Disease (Presence/Absence)
- **Class Distribution**: 
  - Absence (Class 0): 150 samples (55.56%)
  - Presence (Class 1): 120 samples (44.44%)

## Methodology

### 1. Data Preprocessing

- **Train/Test Split**: 70/30 stratified split maintaining class distribution
  - Training Set: 189 samples (105 class 0, 84 class 1)
  - Test Set: 81 samples (45 class 0, 36 class 1)
- **Feature Selection**: 6 key features selected for model training
  - Age, Cholesterol, Blood Pressure (BP), Max Heart Rate (Max HR), ST Depression, Number of Vessels (fluoroscopy)
- **Normalization**: Features standardized using training set statistics (mean=0, std=1)

### 2. Logistic Regression Implementation

**Core Components**:
- **Sigmoid Function**: Maps predictions to probability range [0,1]
- **Cost Function**: Binary cross-entropy with clipping to prevent log(0) errors
- **Gradient Descent**: Iterative optimization with learning rate α=0.01
- **Training Parameters**:
  - Learning Rate (α): 0.01
  - Iterations: 1,000
  - Initial Cost: 0.6916
  - Final Cost: 0.4887

### 3. Model Performance

#### Full Model (6 Features)

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| **Accuracy** | 77.25% | **85.19%** |
| **Precision** | 78.87% | 85.29% |
| **Recall** | 66.67% | 80.56% |
| **F1-Score** | 72.26% | 82.86% |

**Key Observations**:
- No significant overfitting detected (test accuracy > training accuracy)
- Strong generalization capability
- Balanced performance across precision and recall

#### 2D Decision Boundary Analysis

Three feature pairs were analyzed to visualize class separability:

**1. Age vs Cholesterol**
- Test Accuracy: **62.96%**
- F1-Score: 0.50
- Final Cost: 0.672
- **Observation**: Poorest performance among the three pairs, indicating limited separability using only these features

**2. BP vs Max HR**
- Test Accuracy: **72.84%**
- F1-Score: 0.6765
- Final Cost: 0.589
- **Observation**: 10 percentage point improvement over Age-Cholesterol pair, showing better class separation

**3. ST Depression vs Number of Vessels (Fluoroscopy)**
- Test Accuracy: **77.78%**
- F1-Score: 0.7273
- Final Cost: 0.519
- **Observation**: Best 2D performance, approaching full model accuracy, indicating these features are highly discriminative

## Key Results

✅ **Achieved 85.19% test accuracy** with manually implemented logistic regression  
✅ **No overfitting** - model generalizes well to unseen data  
✅ **Balanced metrics** - F1-score of 82.86% indicates good precision-recall balance  
✅ **Feature importance** - ST Depression and Number of Vessels show strongest predictive power  
✅ **Convergence analysis** - Model shows steady cost reduction over 1,000 iterations

## Visualizations

The project includes:
- **Convergence Plot**: Cost function vs iterations showing model optimization
- **Decision Boundaries**: 3 visualizations showing classification regions for different feature pairs
- **Confusion Matrices**: Detailed breakdown of True Positives, True Negatives, False Positives, and False Negatives

## Deployment Status

### Amazon SageMaker Deployment

⚠️ **Status**: Unsuccessful

**Issue**: The deployment to Amazon SageMaker was not completed during this project phase.

**Next Steps for Deployment**:

1. **Model Serialization**
   - Save trained model parameters (weights and bias)
   - Export preprocessing parameters (mean, std)
   - Create model artifact in SageMaker-compatible format

2. **SageMaker Configuration**
   - Create inference script with preprocessing and prediction logic
   - Define model entry point and dependencies
   - Configure instance type and endpoint settings

3. **Deployment Procedure**
   ```python
   # Recommended approach:
   # 1. Package model as .tar.gz with inference.py
   # 2. Upload to S3 bucket
   # 3. Create SageMaker model from artifact
   # 4. Deploy to endpoint with appropriate instance type
   # 5. Test endpoint with sample predictions
   ```

4. **Testing & Validation**
   - Verify endpoint responds correctly
   - Test with sample patient data
   - Monitor latency and performance metrics

## Project Structure

```
logisticRegression-ai-infraestructure/
├── Heart Disease Risk Prediction.ipynb  # Main implementation notebook
├── Heart_Disease_Prediction.csv         # Dataset
└── README.md                             # This file
```

## Technical Implementation Details

### Gradient Descent Optimization

The model uses batch gradient descent with the following update rules:

- **Weight Update**: w = w - α × (1/m) × X^T × (f - y)
- **Bias Update**: b = b - α × (1/m) × Σ(f - y)

Where:
- α = learning rate (0.01)
- m = number of training samples
- f = sigmoid(X × w + b)
- y = actual labels

### Convergence Analysis

- Initial cost: 0.6916
- Final cost: 0.4887
- Cost reduction: 29.3%
- Note: Further iterations may improve convergence (change in last 100 iterations: 0.00186)

## Conclusions

1. **Manual implementation successful**: Logistic regression from scratch achieves competitive performance (85.19% accuracy)

2. **Feature engineering matters**: ST Depression and Number of Vessels are the most predictive features for heart disease

3. **Model generalization**: No overfitting observed, indicating good model design and appropriate regularization through normalization

4. **2D analysis insights**: Even with just 2 features, reasonable accuracy (77.78%) can be achieved with the right feature pair

5. **Deployment readiness**: Model is ready for deployment pending SageMaker configuration


## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib

## Author
David Sarria


