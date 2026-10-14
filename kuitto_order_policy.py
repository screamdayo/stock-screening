"""Pre-entry Kuitto stop prices shared by notifications and forward records."""
import math

SCORE_GAP_CAP_PCT = {0: None, 1: 0.0, 2: 0.25, 3: 0.0, 4: 1.0}
SCORE_STOP_LOSS_PCT = {0: 5.0, 1: 7.0, 2: 7.0, 3: 7.0, 4: 7.0}


def floor_to_valid_tick(price):
    p = float(price)
    limits = [(3000, 1), (5000, 5), (30000, 10), (50000, 50),
              (300000, 100), (500000, 500), (3000000, 1000),
              (5000000, 5000), (30000000, 10000), (50000000, 50000)]
    tick = next((tick for limit, tick in limits if p <= limit), 100000)
    return math.floor((p + 1e-9) / tick) * tick


def preset_stop(signal_close, score):
    score = int(score)
    cap = SCORE_GAP_CAP_PCT.get(score, 0.5)
    pct = SCORE_STOP_LOSS_PCT.get(score, 7.0)
    reference = float(signal_close) if cap is None else floor_to_valid_tick(float(signal_close) * (1 + cap / 100))
    return reference, pct, floor_to_valid_tick(reference * (1 - pct / 100))
