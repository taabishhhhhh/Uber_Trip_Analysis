# 🚖 Uber Trip Demand Forecasting (Jan–Feb 2015)

### 🧠 Objective
Predict daily Uber trip volume using historical trip and vehicle data.

### 📊 Steps Performed
1. Data Cleaning & Exploration (EDA)
2. Feature Engineering (rolling means, lag features, weekend flag)
3. Model Building (Random Forest, XGBoost, Gradient Boosting)
4. Model Evaluation & Explainability (SHAP)
5. Insights & Visualization

### 🧩 Best Model
**Gradient Boosting Regressor**
- MAPE: 7.14%
- RMSE: 1454.74
- R²: 0.983

### 🔍 SHAP Insights
- **Top drivers:** Active vehicles, recent trip trends, lagged trip counts.
- **Interpretation:** Demand rises with more vehicles and consistent past-day trip growth.
- SHAP visuals:
  - ![SHAP Bar](shap_summary_bar.png)
  - ![SHAP Beeswarm](shap_beeswarm.png)

### 💾 Model File
`models/best_model_gradient_boosting.pkl`

### 🚀 Next Steps
- Deploy interactive prediction app with **Streamlit**.
- Incorporate **external features** (weather, holidays) for real-time prediction.
- Build **dashboard in Power BI/Tableau** for live demand monitoring.

---

### 👨‍💻 Author
**Tabish Deshmukh**  
Assistant Director, Deshmukh Computer  
*Passionate Technologist exploring smart solutions, AI tools, and data-driven insights.*

📧 [deshmukhtabish4@gmail.com](mailto:deshmukhtabish4@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/tabish-deshmukh-430279268) | [YouTube](https://youtube.com/@humteendost)
