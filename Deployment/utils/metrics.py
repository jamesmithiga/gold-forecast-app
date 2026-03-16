"""
Metrics Calculation Utilities for Model Evaluation

Calculates performance metrics for machine learning models.
"""

import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: List[float], y_pred: List[float], 
                    model_name: str, ticker: str = 'GC=F') -> Dict[str, Any]:
    """
    Calculate performance metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        ticker: Stock ticker
    
    Returns:
        dict: Performance metrics (RMSE, MAE, MAPE, R2, etc.)
    """
    try:
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE calculation
        nonzero_idx = y_true != 0
        mape = np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100 if np.any(nonzero_idx) else np.nan
        
        r2 = r2_score(y_true, y_pred)
        
        # Directional Accuracy
        diff_true = np.diff(y_true)
        diff_pred = np.diff(y_pred)
        da = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100 if len(diff_true) > 0 else np.nan
        
        metrics = {
            'Ticker': ticker,
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2 Score': r2,
            'Directional Accuracy (%)': da
        }
        
        logger.info(f"{model_name} Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, DA: {da:.1f}%")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise


def calculate_directional_accuracy(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate directional accuracy (percentage of correctly predicted price movements).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        float: Directional accuracy percentage
    """
    try:
        diff_true = np.diff(y_true)
        diff_pred = np.diff(y_pred)
        da = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100 if len(diff_true) > 0 else np.nan
        return da
    except Exception as e:
        logger.error(f"Error calculating directional accuracy: {str(e)}")
        raise


def calculate_volatility(returns: np.ndarray) -> float:
    """
    Calculate volatility (standard deviation of returns).
    
    Args:
        returns: Array of returns
    
    Returns:
        float: Volatility
    """
    try:
        return np.std(returns)
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        raise


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (default: 0.0)
    
    Returns:
        float: Sharpe ratio
    """
    try:
        mean_return = np.mean(returns)
        volatility = calculate_volatility(returns)
        if volatility == 0:
            return 0.0
        return (mean_return - risk_free_rate) / volatility
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        raise


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Array of returns
    
    Returns:
        float: Maximum drawdown
    """
    try:
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown)
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {str(e)}")
        raise


def calculate_beta(returns: np.ndarray, market_returns: np.ndarray) -> float:
    """
    Calculate beta (systematic risk measure).
    
    Args:
        returns: Asset returns
        market_returns: Market returns
    
    Returns:
        float: Beta value
    """
    try:
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        if market_variance == 0:
            return 0.0
        return covariance / market_variance
    except Exception as e:
        logger.error(f"Error calculating beta: {str(e)}")
        raise


def calculate_jensen_alpha(returns: np.ndarray, market_returns: np.ndarray, 
                          risk_free_rate: float = 0.0) -> float:
    """
    Calculate Jensen's alpha.
    
    Args:
        returns: Asset returns
        market_returns: Market returns
        risk_free_rate: Risk-free rate
    
    Returns:
        float: Jensen's alpha
    """
    try:
        beta = calculate_beta(returns, market_returns)
        mean_return = np.mean(returns)
        mean_market_return = np.mean(market_returns)
        return mean_return - (risk_free_rate + beta * (mean_market_return - risk_free_rate))
    except Exception as e:
        logger.error(f"Error calculating Jensen's alpha: {str(e)}")
        raise


def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate information ratio.
    
    Args:
        returns: Asset returns
        benchmark_returns: Benchmark returns
    
    Returns:
        float: Information ratio
    """
    try:
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        if tracking_error == 0:
            return 0.0
        return np.mean(excess_returns) / tracking_error
    except Exception as e:
        logger.error(f"Error calculating information ratio: {str(e)}")
        raise


def calculate_treynor_ratio(returns: np.ndarray, market_returns: np.ndarray, 
                           risk_free_rate: float = 0.0) -> float:
    """
    Calculate Treynor ratio.
    
    Args:
        returns: Asset returns
        market_returns: Market returns
        risk_free_rate: Risk-free rate
    
    Returns:
        float: Treynor ratio
    """
    try:
        beta = calculate_beta(returns, market_returns)
        mean_return = np.mean(returns)
        if beta == 0:
            return 0.0
        return (mean_return - risk_free_rate) / beta
    except Exception as e:
        logger.error(f"Error calculating Treynor ratio: {str(e)}")
        raise


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return).
    
    Args:
        returns: Asset returns
        risk_free_rate: Risk-free rate
    
    Returns:
        float: Sortino ratio
    """
    try:
        mean_return = np.mean(returns)
        downside_returns = returns[returns < mean_return]
        if len(downside_returns) == 0:
            return 0.0
        downside_risk = np.std(downside_returns)
        if downside_risk == 0:
            return 0.0
        return (mean_return - risk_free_rate) / downside_risk
    except Exception as e:
        logger.error(f"Error calculating Sortino ratio: {str(e)}")
        raise


def calculate_calmar_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Calmar ratio.
    
    Args:
        returns: Asset returns
        risk_free_rate: Risk-free rate
    
    Returns:
        float: Calmar ratio
    """
    try:
        mean_return = np.mean(returns)
        max_drawdown = calculate_max_drawdown(returns)
        if max_drawdown == 0:
            return 0.0
        return (mean_return - risk_free_rate) / abs(max_drawdown)
    except Exception as e:
        logger.error(f"Error calculating Calmar ratio: {str(e)}")
        raise


def calculate_value_at_risk(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Asset returns
        confidence_level: Confidence level (default: 0.95)
    
    Returns:
        float: VaR value
    """
    try:
        return np.percentile(returns, (1 - confidence_level) * 100)
    except Exception as e:
        logger.error(f"Error calculating VaR: {str(e)}")
        raise


def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR).
    
    Args:
        returns: Asset returns
        confidence_level: Confidence level
    
    Returns:
        float: CVaR value
    """
    try:
        var = calculate_value_at_risk(returns, confidence_level)
        return np.mean(returns[returns <= var])
    except Exception as e:
        logger.error(f"Error calculating CVaR: {str(e)}")
        raise


def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared value.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        float: R-squared value
    """
    try:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0
        return 1 - (ss_res / ss_tot)
    except Exception as e:
        logger.error(f"Error calculating R-squared: {str(e)}")
        raise


def calculate_adjusted_r_squared(y_true: np.ndarray, y_pred: np.ndarray, 
                                 num_predictors: int) -> float:
    """
    Calculate adjusted R-squared value.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        num_predictors: Number of predictors
    
    Returns:
        float: Adjusted R-squared value
    """
    try:
        n = len(y_true)
        r_squared = calculate_r_squared(y_true, y_pred)
        if n <= num_predictors + 1:
            return r_squared
        return 1 - (1 - r_squared) * (n - 1) / (n - num_predictors - 1)
    except Exception as e:
        logger.error(f"Error calculating adjusted R-squared: {str(e)}")
        raise


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Calculate confusion matrix for classification.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        dict: Confusion matrix counts
    """
    try:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }
    except Exception as e:
        logger.error(f"Error calculating confusion matrix: {str(e)}")
        raise


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        dict: Classification metrics
    """
    try:
        cm = calculate_confusion_matrix(y_true, y_pred)
        
        tp = cm['TP']
        tn = cm['TN']
        fp = cm['FP']
        fn = cm['FN']
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error calculating classification metrics: {str(e)}")
        raise


def get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Get comprehensive regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        dict: Regression metrics (RMSE, MAE, MAPE, R2, etc.)
    """
    try:
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE calculation
        nonzero_idx = y_true != 0
        mape = np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100 if np.any(nonzero_idx) else np.nan
        
        r2 = r2_score(y_true, y_pred)
        
        # Directional Accuracy
        diff_true = np.diff(y_true)
        diff_pred = np.diff(y_pred)
        da = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100 if len(diff_true) > 0 else np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': da
        }
    except Exception as e:
        logger.error(f"Error calculating regression metrics: {str(e)}")
        raise