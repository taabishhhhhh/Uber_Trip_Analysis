Executive Summary

```markdown
# Executive Summary — Uber Trip Forecast (Jan–Feb 2015)

## Objective
Forecast daily Uber trips (short-term) to support driver allocation and reduce unmet demand.

## Dataset
Two months of historical daily data (Jan–Feb 2015) containing:
- `date`, `trips`, `active_vehicles` (primary columns)
- additional derived features created during feature engineering

## Approach
1. Performed EDA to find weekly and seasonal patterns.
2. Engineered time features: day-of-week, weekend flag, month/day, rolling 3/7-day averages, and 1/2/3-day lags.
3. Trained three tree-based regressors with time-ordered train/test split:
   - Random Forest
   - XGBoost
   - Gradient Boosting (best)

## Key Results
- **Best model:** Gradient Boosting Regressor  
- **Performance on holdout test set:**  
  - **MAPE:** 7.67%  
  - **RMSE:** 1,505.94  
  - **R²:** 0.982

These results indicate high day-level predictive accuracy on the provided sample. Rolling means and recent-day lags were consistently the strongest predictors, followed by `active_vehicles`.

## Explainability
- SHAP analysis confirms the model’s reliance on:
  - `active_vehicles` (positive correlation with trips)
  - recent rolling averages (3 and 7 day)
  - previous day(s) trips (lags)
- Visual artifacts (beeswarm, bar plots and force plots) are included in the Reports folder.

## Business impact
- Accurate day-ahead forecasts help optimize driver allocation and reduce wait times or surplus supply. With further per-zone modeling, this system can scale to operational use.

## Next recommended steps
1. **Expand models**: Evaluate Prophet for trend/seasonality and LSTM for longer windows.  
2. **Per-region modeling**: Train per-base or per-zone models.  
3. **Deploy pipeline**: Setup scheduled retraining, monitoring, and a web dashboard for operations.

---

**Prepared by:** Tabish Deshmukh  
**Date:11 November 2025