# Jax Limit Order Book data generation by World Model

This project will be built based on [Learning to simulate realistic limit order book markets from dataas a World Agent](https://dl.acm.org/doi/abs/10.1145/3533271.3561753)

## Motivation
The limit order book data is crucial to a lot of application in finance engineering, but the acqusition of the high-level limit order book data is expensive. For AI agents/ mechine learning algorithm, a vast amount of data is required in their training process, which makes the high quality data more precious. Synetic limit order book data are commonly used for researches, so the building of the algorithms that generates the synthetic data is an important part.

### Why world model
Traditionally, the simulation of market datas was built on multi-agents framework, but the precise data for calibrating each agent is deficient. The world model in the paper generates the market data directly from market's historical data. 

## Things to do
1. market data collection
from the base paper
> In this section we evaluate our world model by comparing the twoproposed approaches in terms of realism and responsiveness. Wetrain our models using NASDAQ TotalView data sent via ITCHprotocol replayed at a simulated exchange at the trading actionlevel. We consider four small-tick stocks, i.e., AVXL, AINV, CNR andAMZN, we use 3 to 4 days of data to train the models, and 9 daysfor testing. The results are averaged for each stock, over the 9 daysperiod. We implement our models extending ABIDES simulator, and we feed real data market from 09:30 to 10:00 to initializethe simulated market, and condition the models.
Our data will include stocks (AAPL, NVDA, ONDS, collected from [NASDAQ ITCH](https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/)), cyptomarket (BTCUSDT, ETHUSDT, collected from [Coinbase API](https://www.coinapi.io/blog/full-order-book-data-in-crypto))

2. Build this world model by Jax
3. Compare the result (Time spending, quality of data) with [JAX_LOB](https://dl.acm.org/doi/abs/10.1145/3604237.3626880)