# BTC 5m Data Collector

Collects tick-level price data for Polymarket BTC 5-minute UP/DOWN markets.

For each 5-minute window, records:
- Polymarket order book (UP/DOWN bid/ask)
- BTC/USD price from the **Chainlink oracle** (the exact feed Polymarket uses for resolution)

Outputs one CSV per market window in `price_collector/data/`.

## Local usage

```bash
pip install -r requirements.txt
python -m price_collector.main --windows 3
```

## Automated collection

A GitHub Actions workflow (`.github/workflows/collect-data.yml`) runs every 6 hours and commits collected CSVs back to the repo.

## Data format

Each row is a tick recorded when the order book changes:

| column | description |
|--------|-------------|
| timestamp | Unix epoch seconds |
| elapsed_sec | Seconds since window open |
| up_bid / up_ask | Best bid/ask for UP token |
| down_bid / down_ask | Best bid/ask for DOWN token |
| up_spread / down_spread | ask - bid |
| btc_price | Chainlink oracle BTC/USD price |

The final row of each CSV contains a `# RESULT` comment with the winner determined from the recorded oracle prices.
