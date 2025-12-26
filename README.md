# Census Data Analysis Project

Two machine learning projects analyzing 1994-1995 U.S. Census Bureau data.

## Projects

### 1. Income Classification (`jpmc_classification.ipynb`)
Binary classifier predicting whether individuals earn >$50K annually.

**Features:**
- Feature engineering (log transforms, binary indicators, categorical collapsing)
- Multiple models evaluated (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
- Imbalance-aware metrics (PR-AUC, balanced accuracy, F1)
- Threshold tuning for business constraints
- Interpretable model with regularized logistic regression

**Best Model:** XGBoost (PR-AUC optimized for class imbalance)

### 2. Customer Segmentation (`jpmc_segmentation.ipynb`)
K-means clustering to identify customer segments for targeted marketing.

**Features:**
- Weighted clustering reflecting population distribution
- 7 distinct customer segments with personas
- Demographic & categorical profiling per segment
- Marketing strategies (product focus, pricing, channels, messaging)
- Customer scoring and deployment guide

**Output:** Production-ready segmentation model with actionable marketing insights

## Data Files
- `census-bureau.data` - Raw census survey data
- `census-bureau.columns` - Column definitions
- `census_fe.data` - Feature-engineered dataset
- `helper.py` - Utility functions for EDA

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Launch Jupyter and open either notebook:
```bash
jupyter notebook jpmc_classification.ipynb
# or
jupyter notebook jpmc_segmentation.ipynb
```

## Key Results

**Classification:** 94% accuracy baseline with 6% high-income class. Model achieves strong PR-AUC through XGBoost and threshold optimization.

**Segmentation:** 7 clusters ranging from "Affluent Professionals" to "Part-Time Workers" with distinct marketing opportunities across demographics and employment patterns.
