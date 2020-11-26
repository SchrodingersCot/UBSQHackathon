# UBSQHackathon (First Round)
UBS Quant Hackathon 2020 - Outperform the S&P 500 Index

Predict stock price returns via neural network.

Useful info:
- Use `alphien.utils.get_all_data` to read the output data

Data Info:
- All market data is inside the `output` directory
- `inclusion.csv` constaints info on which 500 stocks are tradable in the period

## Objective

Description
Act as if you were a US large cap long portfolio manager and needed to build a systematic long only equity stock picking portfolio that performs in all weathers. Use machine learning to help you achieve optimal stock selection with respect to the real-life constraints. Enhanced index strategies are attractive to investors who would like to capture an excess return over a benchmark. The diversified long only portfolio must be equally weighted, fully invested and must contain 50 stocks at any moment of time.

Investment universe
You can only pick stocks which are part of the S&P 500 at the day of selection. Be aware that some stocks can either drop out from the index or enter the index, depending on decisions from the selection committee. Stocks can also drop out from the index if they are de-listed from the exchange, which typically happens following a merger or acquisition event. When a stock drops out of the index and if it’s part of your portfolio, the position has to be closed.

Example:
Netflix can be part of your portfolio starting only from 17 Dec 2010.
Alcoa Inc cannot be part of your portfolio after the 31 Oct 2016.  

Benchmark
The benchmark is the Equally-Weighted S&P 500 index (ticker SPW Index).

Selection criteria
The strategies will be ranked based on the following criteria:
The arithmetic mean of Information ratio (50%), Diversification (30%) and Robustness (20%).
The information ratio will be calculated using the Equally-Weighted S&P 500 index as benchmark.

Backtest period
January 2007 - December 2016
