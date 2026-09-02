# Employee Salary Prediction : Regression Model Comparison

Predicts employee salary from Age, Gender, Education Level, Job Title, and 
Years of Experience, comparing three regression approaches to find the best fit.

## Dataset
[Salary Prediction Dataset](https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer) (Kaggle)

## Models Compared
- Linear Regression
- Decision Tree Regressor (tuned depth via train/test gap analysis)
- Random Forest Regressor (tuned depth via train/test gap analysis)

## Results
| Model                   |    MAE    |    RMSE   |   R²   |
| Linear Regression       | 10,928.92 | 15,782.13 | 0.8961 |
| Decision Tree (depth=4) | 9,959.14  | 13,337.33 | 0.9258 |
| Random Forest (depth=6) | 8,907.61  | 12,217.76 | 0.9377 |

## Key Finding
Random Forest performed best overall. Model depth was tuned by comparing 
training vs test R² to detect overfitting, rather than just picking the 
highest raw accuracy , Random Forest at depth=8 scored marginally higher 
but showed a larger overfitting gap, making depth=6 the better balance.

## Requirements
pandas, scikit-learn, numpy
