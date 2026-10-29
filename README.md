# Smart Canteen: ML Based Food Demand Prediction to Reduce Food Waste

Mini project (Machine Learning, T.E. AI&DS, AVCOE Sangamner).

Predicts how many plates a college canteen will sell in a day from five inputs
(attendance %, temperature, menu rating, exam week, weekend) using Linear Regression
and Random Forest, then a rule-based planner suggests plates to prepare for
breakfast, lunch and snacks.

## Run
```
pip install -r requirements.txt
python smart_canteen.py
```
Prints the dataset checks, model comparison (MAE, RMSE, R2, 5-fold CV R2), feature
importance, planner demo, what-if table and a fixed-policy vs planner comparison.
`output.txt` holds the output of one run; `canteen_dataset_raw.csv` is the generated data.

## Important
The dataset is synthetic (generated in code). To use real canteen records, replace the
dataset block with `pd.read_csv(...)` using the same column names.
This is an educational project; predictions are guidance only.
