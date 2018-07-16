# Stock-Prediction-Comparison
Test performance and reliability of machine learning models from Stacking and Deep Learning for Stock Prediction.

## Models

#### Stacking models
  1. Deep Feed-forward Auto-Encoder Neural Network to reduce dimension + Deep Recurrent Neural Network + ARIMA + Extreme Boosting Gradient Regressor
  2. Adaboost + Bagging + Extra Trees + Gradient Boosting + Random Forest + XGB

#### Deep-learning models
 1. LSTM Recurrent Neural Network
 2. Encoder-Decoder Feed-forward + LSTM Recurrent Neural Network
 3. LSTM Bidirectional Neural Network
 4. 2-Path LSTM Recurrent Neural Network
 5. GRU Recurrent Neural Network
 6. Encoder-Decoder Feed-forward + GRU Recurrent Neural Network
 7. GRU Bidirectional Neural Network
 8. 2-Path GRU Recurrent Neural Network
 9. Vanilla Recurrent Neural Network
 10. Encoder-Decoder Feed-forward + Vanilla Recurrent Neural Network
 11. Vanilla Bidirectional Neural Network
 12. 2-Path Vanilla Recurrent Neural Network
 13. LSTM Sequence-to-Sequence Recurrent Neural Network
 14. LSTM with Attention Recurrent Neural Network
 15. LSTM Sequence-to-Sequence with Attention Recurrent Neural Network
 16. LSTM Sequence-to-Sequence Bidirectional Recurrent Neural Network
 17. LSTM Sequence-to-Sequence with Attention Bidirectional Recurrent Neural Network
 18. LSTM with Attention Scaled-Dot Recurrent Neural Network
 19. LSTM with Dilated Recurrent Neural Network
 20. Only Attention Neural Network
 21. Multihead Attention Neural Network
 22. LSTM with Bahdanau Attention
 23. LSTM with Luong Attention
 24. LSTM with Bahdanau + Luong Attention
 25. DNC Recurrent Neural Network
 26. Residual LSTM Recurrent Neural Network

#### Included simple backtracking on buying and selling decision, simple-investor-arima.ipynb

#### Included stock market study on TESLA stock, tesla-study.ipynb

## Results (not included all)

```text
# buy_stock(pred, df.Close,initial_state=1,delay=4,initial_money=10000,max_buy=3,max_sell=100)
day 0: buy 3 units at price 2306.100036, total balance 7693.899964
day 11, sell 3 units at price 2313.689940, investment 0.329123 %, total balance 10007.589904,
day 21: buy 3 units at price 2251.500000, total balance 7756.089904
day 27, sell 3 units at price 2367.810060, investment 5.165892 %, total balance 10123.899964,
day 37: buy 3 units at price 2374.649964, total balance 7749.250000
day 45, sell 3 units at price 2419.950072, investment 1.907654 %, total balance 10169.200072,
day 54: buy 3 units at price 2457.929994, total balance 7711.270078
day 65, sell 3 units at price 2420.909913, investment -1.506149 %, total balance 10132.179991,
day 77: buy 3 units at price 2485.920045, total balance 7646.259946
day 86, sell 3 units at price 2516.039979, investment 1.211621 %, total balance 10162.299925,
day 97: buy 3 units at price 2443.289979, total balance 7719.009946
day 109, sell 3 units at price 2470.049928, investment 1.095242 %, total balance 10189.059874,
day 120: buy 3 units at price 2622.750000, total balance 7566.309874
day 126, sell 3 units at price 2781.390015, investment 6.048614 %, total balance 10347.699889,
day 144: buy 3 units at price 2900.850036, total balance 7446.849853
day 152, sell 3 units at price 2860.200072, investment -1.401312 %, total balance 10307.049925,
day 158: buy 3 units at price 2878.350036, total balance 7428.699889
day 168, sell 3 units at price 2720.070006, investment -5.498985 %, total balance 10148.769895,
day 180: buy 3 units at price 2941.020081, total balance 7207.749814
day 191, sell 3 units at price 2780.369934, investment -5.462395 %, total balance 9988.119748,
day 201: buy 3 units at price 2774.070006, total balance 7214.049742
day 210, sell 3 units at price 2785.350036, investment 0.406624 %, total balance 9999.399778,
day 218: buy 3 units at price 2760.869934, total balance 7238.529844
day 227, sell 3 units at price 2848.500000, investment 3.174002 %, total balance 10087.029844,
day 241: buy 3 units at price 2978.429994, total balance 7108.599850
day 251, sell 3 units at price 3076.500000, investment 3.292675 %, total balance 10185.099850,

total gained 185.099850, total investment 1.850998 %
```
![alt text](output/arima-investing.png)

LSTM Recurrent Neural Network

![alt text](https://raw.githubusercontent.com/huseinzol05/Stock-Prediction-Comparison/master/output/rnn-only.png)

LSTM Bidirectional Neural Network

![alt text](https://raw.githubusercontent.com/huseinzol05/Stock-Prediction-Comparison/master/output/download%20(1).png)

2-Path LSTM Recurrent Neural Network

![alt text](https://raw.githubusercontent.com/huseinzol05/Stock-Prediction-Comparison/master/output/download.png)

Deep Feed-forward Auto-Encoder Neural Network to reduce dimension + Deep Recurrent Neural Network + ARIMA + Extreme Boosting Gradient Regressor

![alt text](https://raw.githubusercontent.com/huseinzol05/Stock-Prediction-Comparison/master/output/stack-xgb.png)

LSTM Sequence-to-Sequence Recurrent Neural Network

![alt text](output/lstm-seq2seq.png)

LSTM Sequence-to-Sequence with Attention Recurrent Neural Network

![alt text](output/lstm-seq2seq-attention.png)

LSTM Sequence-to-Sequence with Attention Bidirectional Recurrent Neural Network

![alt text](output/lstm-seq2seq-bidirectional-attention.png)

Encoder-Decoder Feed-forward + LSTM Recurrent Neural Network

![alt text](https://raw.githubusercontent.com/huseinzol05/Stock-Prediction-Comparison/master/output/encoder-rnn.png)

Adaboost + Bagging + Extra Trees + Gradient Boosting + Random Forest + XGB

![alt text](https://raw.githubusercontent.com/huseinzol05/Stock-Prediction-Comparison/master/output/stack-ensemble.png)
