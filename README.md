# Sic Bo ML Backend

FastAPI + NumPy + scikit-learn prediction API.

## Deploy on Render.com (Free)

1. GitHub ට push කරන්න
2. render.com → New Web Service
3. ඔබේ repo connect කරන්න
4. Auto-deploy — URL ලැබෙනවා:
   `https://sicbo-ml-api.onrender.com`

## API

### POST /predict
```json
{
  "history": [13, 11, 9, 14, 8, ...],
  "top_n": 6
}
```

Response:
```json
{
  "predictions": [
    {"num": 11, "conf": 87, "small": false, "odd": true, ...},
    ...
  ],
  "meta": {
    "hit_rate": 52,
    "m3_matches": 7,
    "signals": 9
  }
}
```

## Signals (9 total)
1. Transition Matrix (w=5)
2. 3-gram Memory (w=6)
3. 2-gram Memory (w=4)
4. Delta Pattern (w=5)
5. Center Weight (w=3)
6. Arithmetic Progression (w=3)
7. Rolling Window Fuzzy Match (w=8)
8. Sub-sequence Recurrence (w=6)
9. Higher-order Markov via NumPy (w=7)
