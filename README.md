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
Random Forest (max_depth=6) was selected as the best-performing model, but 
the selection process is arguably more valuable than the winning number itself.

**Why not just pick the highest R²?**
A raw accuracy comparison would have naively picked Random Forest at 
max_depth=8, which scored the highest test R² (0.9402). However, checking 
the gap between training R² and test R² for every model revealed that 
max_depth=8 also had the largest overfitting gap of all Random Forest 
variants tested (0.0428) meaning it was starting to memorize training 
data rather than learn purely generalizable patterns.

**The tuning process:**
Each model's depth was tested at multiple values (4, 6, and 8) rather than 
using a single default setting. For the single Decision Tree, this testing 
revealed that a shallower tree (depth=4) actually outperformed a deeper one 
(depth=6) on the test set while reducing the overfitting gap from 0.0517 to 
0.0059 proof that added complexity was hurting more than helping in that 
case. For Random Forest, the trade-off was less clear-cut: deeper trees did 
keep improving test accuracy, but with diminishing returns and a steadily 
growing overfitting gap. Depth=6 was chosen as the balance point: it captured 
nearly all the accuracy available (0.9377 vs 0.9402 at depth=8, a difference 
of just 0.0025) while keeping the overfitting gap meaningfully lower (0.0353 
vs 0.0428).

**Why Random Forest beat Linear Regression at all:**
Examining individual predictions (not just aggregate metrics) showed that 
Linear Regression's largest errors occurred on rows where salary did not 
follow a smooth, linear progression with experience for example, cases 
where a job title or promotion level caused a salary jump that a straight-line 
model couldn't capture. Both tree-based models, which split data using 
threshold-based rules (e.g., "is Job Title = Director?"), handled these 
step-like jumps more accurately, which is reflected in their consistently 
lower RMSE relative to MAE improvement RMSE improved more than MAE did, 
indicating the tree-based models specifically fixed Linear Regression's 
worst individual misses rather than just slightly improving every prediction.

**Takeaway:** The best model wasn't simply "the one with the highest score" 
it was the one that balanced predictive accuracy against evidence of 
overfitting, arrived at through systematic testing rather than a single 
default configuration.
## Requirements
pandas, scikit-learn, numpy
