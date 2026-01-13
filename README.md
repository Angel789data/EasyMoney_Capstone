# EasyMoney_Capstone

## Business Problem
Retail inventory optimization requires accurate demand forecasting. Overstocking increases costs, while stockouts reduce revenue.  
The goal of this project is to predict daily product demand to help inventory planning and operational decisions.

---

## Objective
- Build a robust demand forecasting model using LSTM
- Compare deep learning performance with classical baseline models
- Evaluate impact of predictions on business decisions

---

## Dataset
- Source: internal / simulated retail sales data
- Granularity: daily sales per product
- Period: 2018–2023
- Columns: 
  - `date` → date of sale
  - `product_id` → unique product identifier
  - `sales` → units sold

---

## Methodology
1. **Exploratory Data Analysis (EDA)**
   - Visualize trends, seasonality, outliers
   - Handle missing values and anomalies
2. **Baseline Models**
   - Exponential Smoothing
   - Evaluate using MAE and RMSE
3. **Deep Learning**
   - LSTM model with 7-day sliding window
   - Early stopping and regularization
   - Train/test split with 80/20 ratio
4. **Evaluation**
   - Compare baseline vs LSTM
   - Visualize predictions vs actual sales
   - Analyze business impact

---

## Results
| Model | MAE | RMSE |
|-------|-----|------|
| Exponential Smoothing (Baseline) | 12.8 | 18.2 |
| LSTM | **10.9** | **15.0** |

- LSTM captures seasonality and trends better than baseline  
- Error reduced by ~15%, which can significantly reduce inventory costs

---

## Business Impact
- Lower safety stock requirements  
- Reduced overstock and waste  
- Improved inventory planning and operational efficiency  

---

## Limitations
- LSTM requires more data and longer training time  
- Less interpretable than classical models  
- Gains may not justify complexity for low-volume products

---

## Next Steps
- Hyperparameter tuning with Bayesian optimization  
- Include exogenous variables like promotions, holidays  
- Deploy model as an API for production use

---

## How to Run

### 1️⃣ Clone the repo
```bash
git clone https://github.com/tu_usuario/demand-forecasting-lstm.git
cd demand-forecasting-lstm