"""
First Machine Learning Model: Linear Regression
Author: Vicky-YTZ
Date: 2025
Description: A simple linear regression implementation to understand ML basics
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def generate_sample_data():
    """
    Generate sample data for linear regression
    Returns:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    """
    # Training data
    X_train = np.array([[1], [2], [3], [4], [5]])
    y_train = np.array([2, 4, 6, 8, 10])

    # Test data
    X_test = np.array([[6], [7]])
    y_test = np.array([12, 14])  # Ground truth

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    """
    Train linear regression model
    Args:
        X_train: Training features
        y_train: Training labels
    Returns:
        model: Trained model
    """
    logger.info("Initializing Linear Regression model...")
    model = LinearRegression()

    logger.info("Training model...")
    model.fit(X_train, y_train)

    # Model parameters
    logger.info(f"Model coefficient (slope): {model.coef_[0]:.2f}")
    logger.info(f"Model intercept: {model.intercept_:.2f}")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance
    """
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    logger.info(f"Training MSE: {train_mse:.4f}")
    logger.info(f"Training R2 Score: {train_r2:.4f}")

    # Test predictions
    y_test_pred = model.predict(X_test)

    # Fix: Handle both 1D and 2D arrays
    y_test_pred = y_test_pred.flatten()  # Convert to 1D array
    y_test = np.array(y_test).flatten()  # Convert to 1D array
    X_test = np.array(X_test).flatten()  # Convert to 1D array

    for i, (x, pred, actual) in enumerate(zip(X_test, y_test_pred, y_test)):
        logger.info(f"X={x} -> Prediction: {pred:.2f}, Actual: {actual}")

    return y_train_pred, y_test_pred


def visualize_results(
    X_train, y_train, X_test, y_test, model, y_train_pred, y_test_pred
):
    """
    Create visualization of the model results
    """
    plt.figure(figsize=(10, 6))

    # Training data
    plt.scatter(X_train, y_train, color="blue", label="Training Data", s=100, alpha=0.6)
    plt.plot(X_train, y_train_pred, color="red", linewidth=2, label="Regression Line")

    # Test data
    plt.scatter(
        X_test,
        y_test,
        color="green",
        label="Test Data (Actual)",
        s=100,
        alpha=0.6,
        marker="s",
    )
    plt.scatter(
        X_test,
        y_test_pred,
        color="orange",
        label="Test Data (Predicted)",
        s=100,
        alpha=0.6,
        marker="^",
    )

    plt.xlabel("X (Feature)", fontsize=12)
    plt.ylabel("y (Target)", fontsize=12)
    plt.title("Linear Regression: Training and Test Results", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.savefig("linear_regression_results.png", dpi=300, bbox_inches="tight")
    logger.info("Visualization saved as 'linear_regression_results.png'")
    plt.show()


def main():
    """
    Main execution function
    """
    logger.info("=" * 50)
    logger.info("STARTING LINEAR REGRESSION PROJECT")
    logger.info("=" * 50)

    # 1. Generate data
    X_train, y_train, X_test, y_test = generate_sample_data()
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # 2. Train model
    model = train_model(X_train, y_train)

    # 3. Evaluate model
    y_train_pred, y_test_pred = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 4. Visualize results
    visualize_results(
        X_train, y_train, X_test, y_test, model, y_train_pred, y_test_pred
    )

    logger.info("=" * 50)
    logger.info("PROJECT COMPLETED SUCCESSFULLY")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
