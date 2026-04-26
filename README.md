# Jax Limit Order Book data generation by World Model

This project will be built based on [Bridging the Reality Gap in Limit Order Book Simulation](https://arxiv.org/pdf/2603.24137)(we'll call it base paper below)

## Motivation
The limit order book data is crucial to a lot of application in finance engineering, but the acqusition of the high-level limit order book data is expensive. For AI agents/ mechine learning algorithm, a vast amount of data is required in their training process, which makes the high quality data more precious. Synetic limit order book data are commonly used for researches, so the building of the algorithms that generates the synthetic data is an important part.

## Things to do
1. market data collection
From the base paper, the data should be large-tick assets
Our data will include stocks (INTC, VZ, T and PFE collected from Databento(just like what paper did)), cyptomarket (BTCUSDT, ETHUSDT, collected from [Coinbase API](https://www.coinapi.io/blog/full-order-book-data-in-crypto))

2. Preprocess of data: `src.estimate` for the detail see [code doc](code_base.md) Due: 5/8

3. Model building, Due: 5/21

4. Compare the result (Time spending, quality of data) with [the base paper's code](https://github.com/SaadSouilmi/Queue-Reactive) and [JAX_LOB](https://dl.acm.org/doi/abs/10.1145/3604237.3626880) Due: 5/31