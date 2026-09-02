import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf


sectorMap = {
    "VOO": "Equity", "EEM": "Equity", "VEA": "Equity", "EFA": "Equity", "VWO": "Equity", "FXI": "Equity", "BRK-B": "Equity",
    "XLK": "Tech", "COIN": "Tech", "MSTR": "Tech", "IBIT": "Tech",
    "XLF": "Finance", "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "MS": "Finance", "WFC": "Finance", "V": "Finance", "MA": "Finance", "AXP": "Finance",
    "XLV": "Healthcare", "IBB": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare", "ABBV": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare",
    "XLP": "Consumer", "XLY": "Consumer", "COST": "Consumer", "WMT": "Consumer", "TGT": "Consumer", "HD": "Consumer", "MCD": "Consumer", "SBUX": "Consumer", "NKE": "Consumer", "KO": "Consumer", "PEP": "Consumer",
    "XLE": "Energy", "XLI": "Energy", "XLU": "Energy", "XLB": "Energy", "XLRE": "Energy", "VIS": "Energy", "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "AGG": "Bonds", "BND": "Bonds", "TLT": "Bonds", "IEF": "Bonds", "SHY": "Bonds", "LQD": "Bonds", "HYG": "Bonds", "TIP": "Bonds",
    "GLD": "Commodities", "IAU": "Commodities", "SLV": "Commodities", "USO": "Commodities", "DJP": "Commodities", "SCHH": "REIT", "VNQ": "REIT"
}

sectorLimits = {
    "Commodities": 0.25, 
    "Tech": 0.35,
    "Healthcare": 0.30,
    "Equity": 0.60,
    "Bonds": 0.50
}


def fetch_price_data(tickers, periodYears=5):
    print("Downloading data from Yahoo Finance...")
    rawData = yf.download(tickers, period=f"{periodYears}y", auto_adjust=True, progress=False)["Close"]
    
    if len(tickers) == 1:
        rawData = rawData.to_frame(name=tickers[0])
        
    validCols = []
    for colName in rawData.columns:
        if rawData[colName].isna().mean() < 0.10:
            validCols.append(colName)
            
    stockPrices = rawData[validCols].dropna()
    logReturns = np.log(stockPrices / stockPrices.shift(1)).dropna()
    
    annReturns = logReturns.ewm(span=252 * 2).mean().iloc[-1].values * 252
    
    if hasSklearn:
        shrunkCov, _ = ledoit_wolf(logReturns.values)
        covMatrix = shrunkCov * 252
    else:
        covMatrix = logReturns.cov().values * 252
        
    annVols = logReturns.std().values * np.sqrt(252)
    
    return {
        "prices": stockPrices,
        "ann_returns": annReturns,
        "ann_vols": annVols,
        "cov_matrix": covMatrix,
        "valid_tickers": list(validCols)
    }


def portfolio_return(w, mu):
    return float(np.sum(w * mu))


def portfolio_vol(w, cov):
    return float(np.sqrt(w @ cov @ w))


def sharpe_ratio(w, mu, cov, riskFreeRate):
    portfolioVolatility = portfolio_vol(w, cov)
    if portfolioVolatility > 1e-10:
        return (portfolio_return(w, mu) - riskFreeRate) / portfolioVolatility
    return 0.0


def optimize_max_sharpe(mu, cov, tickers, riskFreeRate=0.045, numMonteCarlo=50000):
    numAssets = len(mu)
    maxIndividualWeight = 0.20
    
    weightConstraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    for sectorName, sectorCap in sectorLimits.items():
        assetIndices = []
        for i in range(len(tickers)):
            if sectorMap.get(tickers[i]) == sectorName:
                assetIndices.append(i)
        if len(assetIndices) > 0:
            weightConstraints.append({
                'type': 'ineq',
                'fun': lambda w, idxList=assetIndices, capVal=sectorCap: capVal - np.sum(w[idxList])
            })
            
    weightMatrix = np.random.dirichlet(np.ones(numAssets), size=numMonteCarlo)
    mcReturns = weightMatrix @ mu
    mcVolatilities = np.sqrt(np.einsum('ij,jk,ik->i', weightMatrix, cov, weightMatrix))
    mcSharpeRatios = (mcReturns - riskFreeRate) / mcVolatilities
    initialWeights = weightMatrix[np.argmax(mcSharpeRatios)]
    
    optResults = minimize(
        fun=lambda w: -sharpe_ratio(w, mu, cov, riskFreeRate),
        x0=initialWeights,
        method="SLSQP",
        bounds=[(0.0, maxIndividualWeight)] * numAssets,
        constraints=weightConstraints,
        options={"ftol": 1e-14, "maxiter": 2000}
    )
    
    optimalWeights = np.clip(optResults.x, 0, maxIndividualWeight)
    optimalWeights = optimalWeights / np.sum(optimalWeights)
    return optimalWeights


def compute_efficient_frontier(mu, cov, numPoints=80):
    numAssets = len(mu)
    targetReturns = np.linspace(mu.min() * 0.95, mu.max(), numPoints)
    frontierVols = []
    frontierReturns = []
    
    for targetVal in targetReturns:
        optResults = minimize(
            fun=lambda w: portfolio_vol(w, cov),
            x0=np.ones(numAssets) / numAssets,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * numAssets,
            constraints=[
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w, targetReturn=targetVal: portfolio_return(w, mu) - targetReturn}
            ],
            options={"ftol": 1e-12, "maxiter": 500}
        )
        if optResults.success:
            normWeights = np.clip(optResults.x, 0, 1)
            normWeights = normWeights / np.sum(normWeights)
            frontierVols.append(portfolio_vol(normWeights, cov))
            frontierReturns.append(portfolio_return(normWeights, mu))
            
    return np.array(frontierVols), np.array(frontierReturns)


def print_results(tickers, assetWeights, mu, assetVols, cov, riskFreeRate):
    optimalReturn = portfolio_return(assetWeights, mu)
    optimalVolatility = portfolio_vol(assetWeights, cov)
    optimalSharpe = sharpe_ratio(assetWeights, mu, cov, riskFreeRate)
    
    print("\n" + "=" * 50)
    print("OPTIMAL PORTFOLIO (Maximum Sharpe Ratio)")
    print("=" * 50)
    print("Sharpe Ratio:     ", round(optimalSharpe, 4))
    print("Expected Return:  ", str(round(optimalReturn * 100, 2)) + "%")
    print("Volatility:       ", str(round(optimalVolatility * 100, 2)) + "%")
    print("Risk-free Rate:   ", str(round(riskFreeRate * 100, 1)) + "%")
    print("-" * 50)
    print("Ticker    Weight    Hist Ret    Hist Vol")
    print("-" * 50)
    
    sortedIndices = np.argsort(assetWeights)[::-1]
    for i in sortedIndices:
        if assetWeights[i] > 0.0005:
            print(f"{tickers[i]:<8}  {assetWeights[i]*100:>6.2f}%    {mu[i]*100:>7.2f}%    {assetVols[i]*100:>7.2f}%")
    print("=" * 50)


def plot_results(tickers, assetWeights, mu, assetVols, cov, riskFreeRate, periodYears):
    optimalReturn = portfolio_return(assetWeights, mu)
    optimalVolatility = portfolio_vol(assetWeights, cov)
    optimalSharpe = sharpe_ratio(assetWeights, mu, cov, riskFreeRate)
    
    frontierVols, frontierReturns = compute_efficient_frontier(mu, cov)
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(frontierVols * 100, frontierReturns * 100, color='blue', linewidth=2, label="Efficient Frontier")
    
    if len(frontierVols) > 0:
        cmlX = np.array([0, max(frontierVols) * 1.3])
        cmlY = (riskFreeRate * 100) + (optimalSharpe * cmlX)
        plt.plot(cmlX, cmlY, color='black', linestyle='--', alpha=0.5, label="Capital Market Line")
        
    for i in range(len(tickers)):
        plt.scatter(assetVols[i] * 100, mu[i] * 100, s=80)
        plt.annotate(tickers[i], (assetVols[i] * 100 + 0.5, mu[i] * 100))
        
    plt.scatter(optimalVolatility * 100, optimalReturn * 100, color='gold', marker='*', s=300, edgecolor='black', zorder=5, label=f"Optimal (Sharpe {round(optimalSharpe, 2)})")
    plt.xlabel("Annual Volatility (%)")
    plt.ylabel("Expected Return (%)")
    plt.title("Risk / Return & Efficient Frontier")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Portfolio Weights Plot
    plt.subplot(1, 2, 2)
    filteredIndices = []
    for i in np.argsort(assetWeights)[::-1]:
        if assetWeights[i] > 0.0005:
            filteredIndices.append(i)
            
    tickerLabels = [tickers[i] for i in filteredIndices]
    weightValues = [assetWeights[i] * 100 for i in filteredIndices]
    
    plt.barh(tickerLabels, weightValues, color='skyblue')
    for barIdx, valNum in enumerate(weightValues):
        plt.text(valNum + 0.5, barIdx, str(round(valNum, 1)) + "%", va='center')
        
    plt.xlabel("Allocation (%)")
    plt.title("Optimal Portfolio Weights")
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("portfolio_optimizer.png")
    plt.show()

