# m9k-proc-unit

Internal data processing pipeline. Experimental module for distributed event stream analysis.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Crypto scanner (runs every 30 min via GitHub Actions)
python scripts/scan_crypto.py

# BIST scanner (runs during market hours via GitHub Actions)
python scripts/scan_bist.py

# Signal tracker (checks pending signals, auto-retrains ML)
python scripts/track_signals.py

# Daily report
python scripts/daily_report.py
```

MIT License
