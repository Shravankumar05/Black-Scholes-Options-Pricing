# Black-Scholes-Options-Pricing

Introduction

- Black-Scholes gives what the price of any derivative should be.
- A financial derivative gives a pay off at a time which is a function of the price.
- A call is the right to buy the stock in the future at a set strike price before the expiry time.
- A put is the right to sell a stock in the future at a set strike price before the expiry time.

Puts and Calls Options Model

- The parameters for the Black-Scholes are:
	- Current stock price, S.
	- Strike price, K.
	- Time to expiration in years, T.
	- Volatility, σ.
	- Risk free interest rate, r.
- We take two parameters as follows:

- Φ(x) is defined as the CDF of a normal distribution with mean 0 and variance 1.
- A call price is found as follows:

- A put price is found as follows:

- Φ(d1) is the probability the that the option will make money and the multiplier gives the expected pay-off overall.
- When volatility is 0 then the formula becomes the forward price difference.

P&L

- P&L value is the difference between the market value of the put or call from the value that you purchased to now:

	- This can be used to get a theoretical future value of the put/call using Black-Scholes model to get the future/current value part essentially becoming the profit/loss.
- We want to build this P&L map to essentially show what future potential values you would need to be in to get a profit and stay out of a loss.
- We can plot a heatmap of how with the strike price and volatility a future call/put price varies from the original price:


Implied volatility

- The implied volatility is given you have the market price what is the volatility that the market is pricing in:

- To obtain this implied volatility we need to use iterations as follows:


	- For the sigma where the function becomes 0 as that is when our "guess" and the actual values are equal.
	- Since this is iterative we need to stop when we are close enough and stop if we are spiralling away to uncontrolled values.

Greeks

- Vega:
	- Is equal to the partial derivative of the difference of the guess value and actual value with respect to volatility:

	- This gives the sensitivity of the option price in relation to volatility and can be found using the normal distribution.
- Delta:

	- The delta is the rate of change of option price with respect to strike price.
	- 
	- This gives how much the option's price moves when underlying stock price changes by $1.
	- This is used for hedging and directional exposure.
- Gamma:

	- Gamma is the partial derivative of the delta with respect to the stock price.
	- This can give the risk of a hedge and the volatility proxy.
- Theta:

	- Is the partial derivative of the option price with respect time.
	- This gives how much the value of the option decays or grows with time.
	- This can be used for selling theta strategies with covered calls.

Technologies

- Python
- Matplotlib
- Numpy
- Streamlit
- Sqlite
- YFinance API