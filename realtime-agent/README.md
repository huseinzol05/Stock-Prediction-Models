## How-to, this model based on [evolution-strategy](https://github.com/huseinzol05/Stock-Prediction-Models/tree/master/agent)

1. You can check [realtime-evolution-strategy.ipynb](realtime-evolution-strategy.ipynb) for to train an evolution strategy to do realtime trading.

I trained the model to learn trading on different stocks,

```python
['TWTR.csv',
 'GOOG.csv',
 'FB.csv',
 'LB.csv',
 'MTDR.csv',
 'CPRT.csv',
 'FSV.csv',
 'TSLA.csv',
 'SINA.csv',
 'GWR.csv']
```

You might want to add more to cover more stochastic patterns.

2. Run [app.py](app.py) to serve the checkpoint model using Flask,

```bash
python3 app.py
```

```text
* Serving Flask app "app" (lazy loading)
* Environment: production
  WARNING: This is a development server. Do not use it in a production deployment.
  Use a production WSGI server instead.
* Debug mode: off
* Running on http://0.0.0.0:8005/ (Press CTRL+C to quit)
```

3. You can check requests example in [request.ipynb](request.ipynb) to get a kickstart.

```bash
curl http://localhost:8005/trade?data=[13.1, 13407500]
```

```python
{'action': 'sell', 'balance': 971.1199990000001, 'investment': '10.224268 %', 'status': 'sell 1 unit, price 16.709999', 'timestamp': '2019-05-26 01:12:10.370206'}
{'action': 'nothing', 'balance': 971.1199990000001, 'status': 'do nothing', 'timestamp': '2019-05-26 01:12:10.376245'}
{'action': 'sell', 'balance': 987.7799990000001, 'investment': '11.066667 %', 'status': 'sell 1 unit, price 16.660000', 'timestamp': '2019-05-26 01:12:10.382282'}
{'action': 'nothing', 'balance': 987.7799990000001, 'status': 'do nothing', 'timestamp': '2019-05-26 01:12:10.388330'}
{'action': 'nothing', 'balance': 987.7799990000001, 'status': 'do nothing', 'timestamp': '2019-05-26 01:12:10.394324'}
{'action': 'sell', 'balance': 1006.1299990000001, 'investment': '18.387097 %', 'status': 'sell 1 unit, price 18.350000', 'timestamp': '2019-05-26 01:12:10.400104'}
{'action': 'nothing', 'balance': 1006.1299990000001, 'status': 'do nothing', 'timestamp': '2019-05-26 01:12:10.405804'}
{'action': 'nothing', 'balance': 1006.1299990000001, 'status': 'do nothing', 'timestamp': '2019-05-26 01:12:10.411531'}
```

## Notes

1. You can use this code to integrate with realtime socket, or any APIs you wanted, imagination is your limit now.
