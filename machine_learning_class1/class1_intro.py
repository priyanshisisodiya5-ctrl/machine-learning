"""Machine Learning Class 1: Introduction to Supervised Learning.

Topic: Understanding how a model learns patterns from data.
Example: Simple Linear Regression (predicting salary based on years of experience).
"""


def simple_linear_regression(x_train: list[float], y_train: list[float]):
    """Calculate slope (m) and intercept (c) for y = m*x + c using Ordinary Least Squares."""
    n = len(x_train)
    x_mean = sum(x_train) / n
    y_mean = sum(y_train) / n

    numerator = sum((x_train[i] - x_mean) * (y_train[i] - y_mean) for i in range(n))
    denominator = sum((x_train[i] - x_mean) ** 2 for i in range(n))

    m = numerator / denominator
    c = y_mean - m * x_mean
    return m, c


def predict(x: float, m: float, c: float) -> float:
    """Predict y given x, slope m, and intercept c."""
    return m * x + c


def mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
    """Compute Mean Squared Error (MSE)."""
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


def main():
    print("=" * 60)
    print("      MACHINE LEARNING - CLASS 1: LINEAR REGRESSION")
    print("=" * 60)

    # Dataset: Years of Experience vs Salary (in thousands USD)
    years_experience = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    salaries = [38.0, 44.0, 52.0, 60.0, 68.0, 75.0, 83.0, 91.0]

    print("[*] Training Data (Experience in years -> Salary in $k):")
    for x, y in zip(years_experience, salaries):
        print(f"    Experience: {x:4.1f} yrs  -->  Salary: ${y:.1f}k")

    # Step 1: Model Training (finding m and c)
    m, c = simple_linear_regression(years_experience, salaries)
    print("\n[*] Model Trained:")
    print(f"    Formula: Salary = ({m:.2f} * Experience) + {c:.2f}")

    # Step 2: Evaluation on training data
    predictions = [predict(x, m, c) for x in years_experience]
    mse = mean_squared_error(salaries, predictions)
    print(f"\n[*] Model Performance:")
    print(f"    Mean Squared Error (MSE): {mse:.4f}")

    # Step 3: Making new predictions
    test_experiences = [2.5, 4.5, 9.0, 10.0]
    print("\n[*] Making New Predictions:")
    for test_x in test_experiences:
        pred_y = predict(test_x, m, c)
        print(f"    Candidate with {test_x:4.1f} yrs experience -> Predicted Salary: ${pred_y:.2f}k")

    print("\n" + "=" * 60)
    print("[*] Class 1 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
