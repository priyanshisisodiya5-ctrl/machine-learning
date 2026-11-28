# Machine Learning

Repository for Machine Learning class notes, practical implementations, datasets, and end-to-end experiments.

---

## 📓 Notebooks

- **[Class 1: Statistics & Dataset Creation](class-1.ipynb)**
  - Dataset generation using `numpy.linspace`
  - Statistical measures (mean, median, mode, variance, standard deviation) using `statistics` & `numpy`
  - Exploratory data analysis and manipulation with `pandas`

- **[Class 2: Correlation Analysis & Feature Scaling](class-2.ipynb)**
  - Correlation matrix computation and visualization using Seaborn (`tips` dataset)
  - Feature transformation & scaling techniques (`MinMaxScaler`, `StandardScaler`)

- **[Class 3: Categorical Encoding & Missing Value Imputation](class-3.ipynb)**
  - Categorical feature encoding (`LabelEncoder`, `OneHotEncoder`)
  - Non-linear transformations (`FunctionTransformer`, `PowerTransformer`)
  - Missing value handling and imputation using `SimpleImputer`

- **[Class 4: Outlier Detection & Column Transformers](class-4.ipynb)**
  - Outlier detection and visualization (Z-score method, IQR technique, boxplots)
  - Advanced feature preprocessing using `ColumnTransformer` (combining scaling and encoding)

- **[Class 5: Introduction to Linear Regression](class-5.ipynb)**
  - Simple Linear Regression implementation using `sklearn.linear_model.LinearRegression`
  - Model fitting, slope coefficient (`m`), intercept (`c`), predictions, and best-fit regression line visualization

- **[Class 6: Linear Regression & Evaluation Metrics](class-6.ipynb)**
  - Regression evaluation metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and Maximum Error (`max_error`)
  - Prediction accuracy and residual error evaluation

- **[Class 7: Multiple Linear Regression with Gradient Descent from Scratch](class-7.ipynb)**
  - Gradient Descent algorithm built from scratch using NumPy
  - Epoch-wise weight updates ($w_1, w_2$) and bias update ($b$) using partial derivatives
  - Cost function (MSE) convergence tracking across epochs

- **[Class 8: Multicollinearity, VIF & Statsmodels Analysis](class-8.ipynb)**
  - Detecting multicollinearity using Variance Inflation Factor (VIF) via `statsmodels`
  - Statistical linear regression analysis using `statsmodels.api` (OLS summary, p-values, and feature selection)

- **[Class 10: Logistic Regression, Hyperparameter Tuning & ROC Analysis](class-10.ipynb)**
  - Binary classification modeling with `LogisticRegression`
  - Hyperparameter tuning and model optimization using `GridSearchCV`
  - Model evaluation: Confusion Matrix, Classification Report (Precision, Recall, F1-Score)
  - Receiver Operating Characteristic (ROC) curve analysis and threshold evaluation

- **[Class 11: End-to-End Loan Prediction Pipeline & Model Serialization](class-11.ipynb)**
  - Complete data cleaning, preprocessing, and exploratory data analysis on the loan dataset
  - Feature encoding, scaling, train-test splitting, and Logistic Regression model training
  - Model persistence and deployment preparation via `joblib` (`logistic_regression.pkl`)

- **[Class 12: K-Nearest Neighbors (KNN) & Cross-Validation](class-12.ipynb)**
  - Supervised classification using `KNeighborsClassifier`
  - Hyperparameter tuning across neighbor counts (`k`) and weighting schemes
  - Model validation using `StratifiedKFold` cross-validation and `cross_val_score`

---

## 📂 Datasets & Models

- **[`bikes.csv`](bikes.csv)** - Dataset of used bikes with specifications, mileage, power, and pricing.
- **[`Loan_default.csv`](Loan_default.csv)** - Comprehensive loan default prediction dataset.
- **[`cleaned_loan_dataset.csv`](cleaned_loan_dataset.csv)** - Cleaned and preprocessed dataset for loan eligibility prediction.
- **[`logistic_regression.pkl`](logistic_regression.pkl)** - Serialized, pre-trained Logistic Regression model saved with `joblib`.

---

*Maintained by [priyanshisisodiya5-ctrl](https://github.com/priyanshisisodiya5-ctrl)*
