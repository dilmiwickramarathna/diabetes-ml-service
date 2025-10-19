## v0.1
- Implemented baseline LinearRegression model.
- Added StandardScaler preprocessing.
- Built FastAPI service with /health and /predict endpoints.
- Dockerized application.

## v0.2
- Switched from LinearRegression to Ridge Regression.
- Added feature selection (top 8 features).
- Optional high-risk flag with threshold=200.
- RMSE improved: 62.1 → 58.3
- High-risk flag precision=0.75, recall=0.80
- Justification: Ridge reduces overfitting, feature selection reduces noise, threshold helps prioritize patients for follow-up.
