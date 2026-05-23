import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_prediction_chart(dates: list, actual_sales: list, predicted_sales: list, model_name: str, output_path: str = "prediction_chart.png"):
    """
    Generates a chart comparing actual vs predicted sales for a single model.
    """
    plt.figure(figsize=(12, 6))
    if len(actual_sales) == len(dates):
        plt.plot(dates, actual_sales, label='Actual Sales', alpha=0.7, color='black')
    plt.plot(dates, predicted_sales, label=f'{model_name} Predicted Sales', alpha=0.7)
    plt.title(f'Actual Sales vs Predicted Sales ({model_name})')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_model_comparison_chart(dates: list, actual_sales: list, model_predictions: dict, output_path: str = "model_comparison.png"):
    """
    Generates a chart comparing actual vs multiple models' predictions.
    """
    plt.figure(figsize=(16, 8))
    plt.plot(dates, actual_sales, label='Actual Sales', color='black', linewidth=2)
    
    for model_name, predictions in model_predictions.items():
        plt.plot(dates, predictions, label=model_name, alpha=0.8)
        
    plt.title('Model Comparison: Daily Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
