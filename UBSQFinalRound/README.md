# UBSQHackathon (Final Round)

UBS Quant Hackathon 2020 - Enhanced FX Strategy

## Objective
Description
Act as if you were a FX portfolio manager willing to build a diversified currency trading strategy. The strategy will aim at capturing "carry", i.e. taking advantage of interest rates spreads between currencies, while controlling the volatility of the FX portfolio.


Investment universe
The Investment Universe is composed of 22 (properly rolled) FX futures. The universe can be called using the function getTickersFXCarryStrategy.

> getTickersFXCarryStrategy()
         ticker                            description baseCurrency quoteCurrency
1     AD Curncy                      Australian Dollar          AUD           USD
2  AD-CD Curncy    Australian Dollar / Canadian Dollar          AUD           CAD
3  AD-JY Curncy       Australian Dollar / Japanese Yen          AUD           JPY
4  AD-NV Curncy Australian Dollar / New Zealand Dollar          AUD           NZD
5     BP Curncy                          British Pound          GBP           USD
6  BP-JY Curncy           British Pound / Japanese Yen          GBP           JPY
7  BP-SF Curncy            British Pound / Swiss Franc          GBP           CHF
8     BR Curncy                         Brazilian Real          BRL           USD
9  BR-JY Curncy          Brazilian Real / Japanese Yen          BRL           JPY
10 BR-PE Curncy          Brazilian Real / Mexican Peso          BRL           MXN
11    CD Curncy                        Canadian Dollar          CAD           USD
12 CD-NV Curncy   Canadian Dollar / New Zealand Dollar          CAD           NZD
13    EC Curncy                            Euro Dollar          EUR           USD
14 EC-AD Curncy               Euro / Australian Dollar          EUR           AUD
15 EC-JY Curncy                    Euro / Japanese Yen          EUR           JPY
16    JY Curncy                           Japanese Yen          JPY           USD
17    NV Curncy                     New Zealand Dollar          NZD           USD
18 NV-JY Curncy      New Zealand Dollar / Japanese Yen          NZD           JPY
19    PE Curncy                           Mexican Peso          MXN           USD
20    RF Curncy                     Euro / Swiss Franc          EUR           CHF
21    SE Curncy                          Swedish Krona          SEK           USD
22    SF Curncy                            Swiss Franc          CHF           USD
Example:
To be long Australian dollar and short US Dollar: buy the ticker AD.

To be short Canadian Dollar and long US Dollar: sell the ticker CD.  

Benchmark
No benchmark.

Strategy definitions
A Strategy is a portfolio of Investable FX Indices. Its weights are managed dynamically on the basis of an algorithm which generates transaction signals.

The strategy must comply with the following constraints:

Each weight must be in a range of [-100%; +100%]
The portfolio's annualised volatility must lie between 8% and 12% for the entire 2007::2016 period.
First weights have to be generated before/on 2008-01-02 (1 year calibration/training. Rolling/expanding windows are allowed)
Payout has to be replicable and contain no lookahead bias
All time series data has to be called via the DataFeatures object - no data calls in your payout.
Whole notebook has to be ran under 15 mins (excluding the testFXCarrySubmission function). Training of models/processing of data inclusive.
For the avoidance of doubt, the number of Investable FX Indices that shall be used within the Strategy is subject to the team's discretion. The right balance between leverage and volatility is key here.

Strategy transactions
Transaction costs: No transaction costs.

Transaction frequency: The maximum transaction frequency is daily (no intraday trades). All trades are executed at market close prices.

Transaction signals: Transaction signals must be based on data available at close-of-business on the day prior to the execution.

Backtest period
The in-sample Backtest Period runs from January 2007 to December 2016.
