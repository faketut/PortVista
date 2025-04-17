import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from scipy import stats
from datetime import datetime

class PortfolioRiskSimulator:
    def __init__(self, db_path=None):
        """
        Initialize the Portfolio Risk Simulator
        
        Parameters:
        -----------
        db_path : str, optional
            Path to SQLite database containing portfolio data
        """
        self.db_path = db_path
        self.portfolio = None
        self.returns = None
        self.weights = None
        self.cov_matrix = None
        self.sim_results = None
        self.confidence_level = 0.95
        
    def load_portfolio_from_db(self, query):
        """
        Load portfolio data from SQL database
        
        Parameters:
        -----------
        query : str
            SQL query to extract portfolio data
            
        Returns:
        --------
        DataFrame with portfolio data
        """
        if self.db_path is None:
            raise ValueError("Database path not specified")
        
        try:
            conn = sqlite3.connect(self.db_path)
            self.portfolio = pd.read_sql(query, conn)
            conn.close()
            print(f"Successfully loaded portfolio with {len(self.portfolio)} records")
            return self.portfolio
        except Exception as e:
            print(f"Error loading from database: {e}")
            return None
    
    def load_portfolio_from_csv(self, file_path):
        """
        Load portfolio data from CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to CSV file with portfolio data
            
        Returns:
        --------
        DataFrame with portfolio data
        """
        try:
            self.portfolio = pd.read_csv(file_path)
            print(f"Successfully loaded portfolio with {len(self.portfolio)} records")
            return self.portfolio
        except Exception as e:
            print(f"Error loading from CSV: {e}")
            return None
    
    def calculate_returns(self, price_column='price', date_column='date', ticker_column='ticker'):
        """
        Calculate daily returns from price data
        
        Parameters:
        -----------
        price_column : str
            Column name containing price data
        date_column : str
            Column name containing date data
        ticker_column : str
            Column name containing asset identifier
            
        Returns:
        --------
        DataFrame with return data
        """
        if self.portfolio is None:
            raise ValueError("Portfolio data not loaded")
            
        # Convert date column to datetime if it's not already
        self.portfolio[date_column] = pd.to_datetime(self.portfolio[date_column])
        
        # Pivot the data to have tickers as columns and dates as index
        prices = self.portfolio.pivot(index=date_column, columns=ticker_column, values=price_column)
        
        # Calculate daily returns
        self.returns = prices.pct_change().dropna()
        
        return self.returns
    
    def set_portfolio_weights(self, weights=None):
        """
        Set portfolio weights
        
        Parameters:
        -----------
        weights : dict or array-like
            Weights for each asset in the portfolio. If None, equal weights are assumed.
        """
        if self.returns is None:
            raise ValueError("Returns data not calculated")
            
        assets = self.returns.columns
        
        if weights is None:
            # Equal weights
            self.weights = np.array([1/len(assets)] * len(assets))
        elif isinstance(weights, dict):
            # Dictionary of weights
            self.weights = np.array([weights.get(asset, 0) for asset in assets])
        else:
            # Array-like weights
            self.weights = np.array(weights)
            
        # Normalize weights to sum to 1
        self.weights = self.weights / np.sum(self.weights)
        
        return self.weights
    
    def calculate_covariance_matrix(self, annualized=True, trading_days=252):
        """
        Calculate the covariance matrix of returns
        
        Parameters:
        -----------
        annualized : bool
            Whether to annualize the covariance matrix
        trading_days : int
            Number of trading days in a year
            
        Returns:
        --------
        Covariance matrix
        """
        if self.returns is None:
            raise ValueError("Returns data not calculated")
            
        self.cov_matrix = self.returns.cov()
        
        if annualized:
            self.cov_matrix = self.cov_matrix * trading_days
            
        return self.cov_matrix
    
    def run_monte_carlo_simulation(self, num_simulations=10000, time_horizon=252, seed=None):
        """
        Run Monte Carlo simulation for portfolio returns
        
        Parameters:
        -----------
        num_simulations : int
            Number of simulation paths
        time_horizon : int
            Number of days to simulate
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Array of simulated portfolio values
        """
        if self.returns is None or self.weights is None or self.cov_matrix is None:
            raise ValueError("Returns, weights, or covariance matrix not set")
            
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Calculate mean returns
        mean_returns = self.returns.mean().values
        
        # Number of assets
        num_assets = len(self.weights)
        
        # Cholesky decomposition for correlated random variables
        cholesky_matrix = np.linalg.cholesky(self.cov_matrix)
        
        # Initialize simulation results array
        self.sim_results = np.zeros((num_simulations, time_horizon))
        
        # Run simulations
        for i in range(num_simulations):
            # Initial portfolio value
            portfolio_value = 1
            
            for t in range(time_horizon):
                # Generate random standard normal variables
                rand_vars = np.random.standard_normal(num_assets)
                
                # Generate correlated random returns
                correlated_returns = mean_returns + np.dot(cholesky_matrix, rand_vars)
                
                # Calculate portfolio return
                portfolio_return = np.sum(self.weights * correlated_returns)
                
                # Update portfolio value
                portfolio_value *= (1 + portfolio_return)
                
                # Store the result
                self.sim_results[i, t] = portfolio_value
                
        return self.sim_results
    
    def calculate_historical_var(self, confidence_level=None):
        """
        Calculate historical Value at Risk (VaR)
        
        Parameters:
        -----------
        confidence_level : float, optional
            Confidence level for VaR calculation (default is class attribute)
            
        Returns:
        --------
        Historical VaR value
        """
        if self.returns is None or self.weights is None:
            raise ValueError("Returns or weights not set")
            
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        # Calculate portfolio returns
        portfolio_returns = np.sum(self.returns * self.weights, axis=1)
        
        # Calculate VaR
        var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        
        return var
    
    def calculate_parametric_var(self, confidence_level=None, time_horizon=1):
        """
        Calculate parametric (variance-covariance) Value at Risk (VaR)
        
        Parameters:
        -----------
        confidence_level : float, optional
            Confidence level for VaR calculation (default is class attribute)
        time_horizon : int
            Time horizon in days
            
        Returns:
        --------
        Parametric VaR value
        """
        if self.returns is None or self.weights is None or self.cov_matrix is None:
            raise ValueError("Returns, weights, or covariance matrix not set")
            
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        # Calculate portfolio mean and standard deviation
        mean_returns = self.returns.mean().values
        portfolio_mean = np.sum(self.weights * mean_returns) * time_horizon
        portfolio_std = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)) * time_horizon)
        
        # Calculate z-score
        z_score = stats.norm.ppf(1 - confidence_level)
        
        # Calculate VaR
        var = -portfolio_mean + z_score * portfolio_std
        
        return var
    
    def calculate_simulation_var(self, confidence_level=None, time_horizon=None):
        """
        Calculate simulation-based Value at Risk (VaR)
        
        Parameters:
        -----------
        confidence_level : float, optional
            Confidence level for VaR calculation (default is class attribute)
        time_horizon : int, optional
            Time horizon for which to calculate VaR
            
        Returns:
        --------
        Simulation-based VaR value
        """
        if self.sim_results is None:
            raise ValueError("Monte Carlo simulation not run")
            
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        if time_horizon is None:
            # Default to end of simulation
            time_horizon = self.sim_results.shape[1] - 1
            
        # Get final portfolio values
        final_values = self.sim_results[:, time_horizon]
        
        # Calculate returns relative to initial value
        returns = final_values - 1
        
        # Calculate VaR
        var = -np.percentile(returns, 100 * (1 - confidence_level))
        
        return var
    
    def calculate_historical_cvar(self, confidence_level=None):
        """
        Calculate historical Conditional Value at Risk (CVaR) / Expected Shortfall
        
        Parameters:
        -----------
        confidence_level : float, optional
            Confidence level for CVaR calculation (default is class attribute)
            
        Returns:
        --------
        Historical CVaR value
        """
        if self.returns is None or self.weights is None:
            raise ValueError("Returns or weights not set")
            
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        # Calculate portfolio returns
        portfolio_returns = np.sum(self.returns * self.weights, axis=1)
        
        # Calculate VaR
        var = -np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        
        # Calculate CVaR
        cvar = -portfolio_returns[portfolio_returns <= -var].mean()
        
        return cvar
    
    def calculate_simulation_cvar(self, confidence_level=None, time_horizon=None):
        """
        Calculate simulation-based Conditional Value at Risk (CVaR)
        
        Parameters:
        -----------
        confidence_level : float, optional
            Confidence level for CVaR calculation (default is class attribute)
        time_horizon : int, optional
            Time horizon for which to calculate CVaR
            
        Returns:
        --------
        Simulation-based CVaR value
        """
        if self.sim_results is None:
            raise ValueError("Monte Carlo simulation not run")
            
        if confidence_level is None:
            confidence_level = self.confidence_level
            
        if time_horizon is None:
            # Default to end of simulation
            time_horizon = self.sim_results.shape[1] - 1
            
        # Get final portfolio values
        final_values = self.sim_results[:, time_horizon]
        
        # Calculate returns relative to initial value
        returns = final_values - 1
        
        # Calculate VaR
        var = -np.percentile(returns, 100 * (1 - confidence_level))
        
        # Calculate CVaR
        cvar = -returns[returns <= -var].mean()
        
        return cvar
    
    def plot_simulation_paths(self, num_paths=100, title="Monte Carlo Simulation Paths"):
        """
        Plot a subset of Monte Carlo simulation paths
        
        Parameters:
        -----------
        num_paths : int
            Number of paths to plot
        title : str
            Plot title
            
        Returns:
        --------
        Matplotlib figure
        """
        if self.sim_results is None:
            raise ValueError("Monte Carlo simulation not run")
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot a subset of paths
        for i in range(min(num_paths, self.sim_results.shape[0])):
            ax.plot(self.sim_results[i], alpha=0.3, linewidth=0.5)
            
        # Plot the mean path
        mean_path = np.mean(self.sim_results, axis=0)
        ax.plot(mean_path, color='red', linewidth=2, label='Mean Path')
        
        # Add percentiles
        percentile_5 = np.percentile(self.sim_results, 5, axis=0)
        percentile_95 = np.percentile(self.sim_results, 95, axis=0)
        
        ax.plot(percentile_5, color='black', linestyle='--', linewidth=1.5, label='5th Percentile')
        ax.plot(percentile_95, color='black', linestyle='--', linewidth=1.5, label='95th Percentile')
        
        ax.set_title(title)
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_var_distribution(self, time_horizon=None, title="Portfolio Value Distribution"):
        """
        Plot the distribution of final portfolio values and mark VaR
        
        Parameters:
        -----------
        time_horizon : int, optional
            Time horizon for which to plot distribution
        title : str
            Plot title
            
        Returns:
        --------
        Matplotlib figure
        """
        if self.sim_results is None:
            raise ValueError("Monte Carlo simulation not run")
            
        if time_horizon is None:
            # Default to end of simulation
            time_horizon = self.sim_results.shape[1] - 1
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get final portfolio values
        final_values = self.sim_results[:, time_horizon]
        
        # Calculate returns relative to initial value
        returns = final_values - 1
        
        # Calculate VaR
        var = -np.percentile(returns, 100 * (1 - self.confidence_level))
        
        # Plot distribution
        sns.histplot(returns, bins=50, kde=True, ax=ax)
        
        # Mark VaR
        ax.axvline(-var, color='red', linestyle='--', linewidth=2, 
                   label=f'VaR {self.confidence_level*100:.1f}%: {var:.2%}')
        
        # Mark CVaR
        cvar = -returns[returns <= -var].mean()
        ax.axvline(-cvar, color='purple', linestyle='--', linewidth=2,
                  label=f'CVaR {self.confidence_level*100:.1f}%: {cvar:.2%}')
        
        ax.set_title(title)
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_var_comparison(self):
        """
        Plot comparison of VaR methods
        
        Returns:
        --------
        Matplotlib figure
        """
        if self.returns is None or self.weights is None or self.cov_matrix is None or self.sim_results is None:
            raise ValueError("Data missing for VaR comparison")
            
        confidence_levels = np.arange(0.9, 0.995, 0.01)
        historical_vars = []
        parametric_vars = []
        simulation_vars = []
        
        for cl in confidence_levels:
            historical_vars.append(self.calculate_historical_var(cl))
            parametric_vars.append(self.calculate_parametric_var(cl))
            simulation_vars.append(self.calculate_simulation_var(cl))
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(confidence_levels * 100, historical_vars, 'o-', label='Historical VaR')
        ax.plot(confidence_levels * 100, parametric_vars, 's-', label='Parametric VaR')
        ax.plot(confidence_levels * 100, simulation_vars, '^-', label='Simulation VaR')
        
        ax.set_title('VaR Comparison Across Methods')
        ax.set_xlabel('Confidence Level (%)')
        ax.set_ylabel('Value at Risk')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def generate_risk_report(self, output_format='text'):
        """
        Generate a comprehensive risk report
        
        Parameters:
        -----------
        output_format : str
            Format of the report ('text' or 'html')
            
        Returns:
        --------
        Risk report in specified format
        """
        if self.returns is None or self.weights is None or self.cov_matrix is None:
            raise ValueError("Required data missing for risk report")
            
        # Calculate portfolio statistics
        portfolio_returns = np.sum(self.returns * self.weights, axis=1)
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility
        
        # Calculate VaR using different methods
        historical_var = self.calculate_historical_var()
        parametric_var = self.calculate_parametric_var()
        
        simulation_var = None
        simulation_cvar = None
        if self.sim_results is not None:
            simulation_var = self.calculate_simulation_var()
            simulation_cvar = self.calculate_simulation_cvar()
            
        # Calculate historical CVaR
        historical_cvar = self.calculate_historical_cvar()
        
        # Create report
        if output_format == 'text':
            report = []
            report.append("=" * 60)
            report.append("PORTFOLIO RISK REPORT")
            report.append("=" * 60)
            report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Confidence Level: {self.confidence_level:.2%}")
            report.append("\nPORTFOLIO STATISTICS")
            report.append("-" * 60)
            report.append(f"Number of Assets: {len(self.weights)}")
            report.append(f"Annual Return: {annual_return:.2%}")
            report.append(f"Annual Volatility: {annual_volatility:.2%}")
            report.append(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            
            report.append("\nVALUE AT RISK (VaR)")
            report.append("-" * 60)
            report.append(f"Historical VaR ({self.confidence_level:.2%}): {historical_var:.2%}")
            report.append(f"Parametric VaR ({self.confidence_level:.2%}): {parametric_var:.2%}")
            if simulation_var is not None:
                report.append(f"Simulation VaR ({self.confidence_level:.2%}): {simulation_var:.2%}")
                
            report.append("\nCONDITIONAL VALUE AT RISK (CVaR)")
            report.append("-" * 60)
            report.append(f"Historical CVaR ({self.confidence_level:.2%}): {historical_cvar:.2%}")
            if simulation_cvar is not None:
                report.append(f"Simulation CVaR ({self.confidence_level:.2%}): {simulation_cvar:.2%}")
                
            report.append("\nPORTFOLIO WEIGHTS")
            report.append("-" * 60)
            for i, asset in enumerate(self.returns.columns):
                report.append(f"{asset}: {self.weights[i]:.2%}")
                
            return "\n".join(report)
        
        else:  # HTML format
            # Simple HTML format for demonstration
            report = []
            report.append("<html><head><title>Portfolio Risk Report</title>")
            report.append("<style>body{font-family:Arial;margin:20px}")
            report.append("table{border-collapse:collapse;width:100%}")
            report.append("th,td{border:1px solid #ddd;padding:8px;text-align:left}")
            report.append("th{background-color:#f2f2f2}")
            report.append("h1,h2{color:#333}</style></head><body>")
            
            report.append("<h1>PORTFOLIO RISK REPORT</h1>")
            report.append(f"<p><b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>")
            report.append(f"<b>Confidence Level:</b> {self.confidence_level:.2%}</p>")
            
            report.append("<h2>PORTFOLIO STATISTICS</h2>")
            report.append("<table>")
            report.append("<tr><th>Metric</th><th>Value</th></tr>")
            report.append(f"<tr><td>Number of Assets</td><td>{len(self.weights)}</td></tr>")
            report.append(f"<tr><td>Annual Return</td><td>{annual_return:.2%}</td></tr>")
            report.append(f"<tr><td>Annual Volatility</td><td>{annual_volatility:.2%}</td></tr>")
            report.append(f"<tr><td>Sharpe Ratio</td><td>{sharpe_ratio:.2f}</td></tr>")
            report.append("</table>")
            
            report.append("<h2>VALUE AT RISK (VaR)</h2>")
            report.append("<table>")
            report.append("<tr><th>Method</th><th>Value</th></tr>")
            report.append(f"<tr><td>Historical VaR ({self.confidence_level:.2%})</td><td>{historical_var:.2%}</td></tr>")
            report.append(f"<tr><td>Parametric VaR ({self.confidence_level:.2%})</td><td>{parametric_var:.2%}</td></tr>")
            if simulation_var is not None:
                report.append(f"<tr><td>Simulation VaR ({self.confidence_level:.2%})</td><td>{simulation_var:.2%}</td></tr>")
            report.append("</table>")
            
            report.append("<h2>CONDITIONAL VALUE AT RISK (CVaR)</h2>")
            report.append("<table>")
            report.append("<tr><th>Method</th><th>Value</th></tr>")
            report.append(f"<tr><td>Historical CVaR ({self.confidence_level:.2%})</td><td>{historical_cvar:.2%}</td></tr>")
            if simulation_cvar is not None:
                report.append(f"<tr><td>Simulation CVaR ({self.confidence_level:.2%})</td><td>{simulation_cvar:.2%}</td></tr>")
            report.append("</table>")
            
            report.append("<h2>PORTFOLIO WEIGHTS</h2>")
            report.append("<table>")
            report.append("<tr><th>Asset</th><th>Weight</th></tr>")
            for i, asset in enumerate(self.returns.columns):
                report.append(f"<tr><td>{asset}</td><td>{self.weights[i]:.2%}</td></tr>")
            report.append("</table>")
            
            report.append("</body></html>")
            return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    np.random.seed(42)
    
    # Create price data for a sample portfolio
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
    assets = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    # Initialize price at 100
    prices = np.ones((len(dates), len(assets))) * 100
    
    # Generate random daily returns with correlation
    mu = np.array([0.0002, 0.0001, 0.0003, 0.0002, 0.0001])  # Mean daily returns
    sigma = np.array([0.015, 0.012, 0.018, 0.014, 0.020])    # Daily volatility
    
    # Correlation matrix
    corr = np.array([
        [1.00, 0.70, 0.60, 0.65, 0.55],
        [0.70, 1.00, 0.55, 0.60, 0.50],
        [0.60, 0.55, 1.00, 0.70, 0.65],
        [0.65, 0.60, 0.70, 1.00, 0.60],
        [0.55, 0.50, 0.65, 0.60, 1.00]
    ])
    
    # Convert correlation to covariance
    cov = np.outer(sigma, sigma) * corr
    
    # Generate random returns
    daily_returns = np.random.multivariate_normal(mu, cov, len(dates))
    
    # Update prices based on returns
    for i in range(1, len(dates)):
        prices[i] = prices[i-1] * (1 + daily_returns[i])
    
    # Create a DataFrame
    data = []
    for i, date in enumerate(dates):
        for j, asset in enumerate(assets):
            data.append({
                'date': date,
                'ticker': asset,
                'price': prices[i, j]
            })
    
    sample_data = pd.DataFrame(data)
    
    # Save to CSV for demo
    sample_data.to_csv('sample_portfolio_data.csv', index=False)
    
    # Initialize simulator
    simulator = PortfolioRiskSimulator()
    
    # Load data
    simulator.load_portfolio_from_csv('sample_portfolio_data.csv')
    
    # Calculate returns
    simulator.calculate_returns()
    
    # Set portfolio weights (equal weights)
    simulator.set_portfolio_weights()
    
    # Calculate covariance matrix
    simulator.calculate_covariance_matrix()
    
    # Run Monte Carlo simulation
    simulator.run_monte_carlo_simulation(num_simulations=10000, time_horizon=252)
    
    # Calculate VaR and CVaR
    historical_var = simulator.calculate_historical_var()
    parametric_var = simulator.calculate_parametric_var()
    simulation_var = simulator.calculate_simulation_var()
    
    historical_cvar = simulator.calculate_historical_cvar()
    simulation_cvar = simulator.calculate_simulation_cvar()
    
    print(f"Historical VaR (95%): {historical_var:.2%}")
    print(f"Parametric VaR (95%): {parametric_var:.2%}")
    print(f"Simulation VaR (95%): {simulation_var:.2%}")
    print(f"Historical CVaR (95%): {historical_cvar:.2%}")
    print(f"Simulation CVaR (95%): {simulation_cvar:.2%}")
    
    # Generate plots
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    simulator.plot_simulation_paths()
    
    plt.subplot(2, 2, 2)
    simulator.plot_var_distribution()
    
    plt.subplot(2, 2, 3)
    simulator.plot_var_comparison()
    
    plt.tight_layout()
    plt.savefig('portfolio_risk_analysis.png')
    
    # Generate risk report
    risk_report = simulator.generate_risk_report()
    print("\n" + risk_report)