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

#### Included simple backtracking on buying and selling decision, simple-investor-arima.ipynb


## Results (not included all)

```text
day 0: buy 1 units at price 768.700012, total balance 231.299988
day 11, sell 1 units at price 771.229980, investment 0.329123 %, total balance 1002.529968,
day 21: buy 1 units at price 768.700012, total balance 233.829956
day 27, sell 2 units at price 1578.540040, investment 5.165892 %, total balance 1812.369996,
day 37: buy 2 units at price 1537.400024, total balance 274.969972
day 45, sell 4 units at price 3226.600096, investment 1.907654 %, total balance 3501.570068,
day 54: buy 3 units at price 2306.100036, total balance 1195.470032
day 65, sell 7 units at price 5648.789797, investment -1.506149 %, total balance 6844.259829,
day 77: buy 3 units at price 2306.100036, total balance 4538.159793
day 86, sell 10 units at price 8386.799930, investment 1.211621 %, total balance 12924.959723,
day 97: buy 3 units at price 2306.100036, total balance 10618.859687
day 109, sell 13 units at price 10703.549688, investment 1.095242 %, total balance 21322.409375,
day 120: buy 3 units at price 2306.100036, total balance 19016.309339
day 126, sell 16 units at price 14834.080080, investment 6.048614 %, total balance 33850.389419,
day 144: buy 3 units at price 2306.100036, total balance 31544.289383
day 152, sell 19 units at price 18114.600456, investment -1.401312 %, total balance 49658.889839,
day 158: buy 3 units at price 2306.100036, total balance 47352.789803
day 168, sell 22 units at price 19947.180044, investment -5.498985 %, total balance 67299.969847,
day 180: buy 3 units at price 2306.100036, total balance 64993.869811
day 191, sell 25 units at price 23169.749450, investment -5.462395 %, total balance 88163.619261,
day 201: buy 3 units at price 2306.100036, total balance 85857.519225
day 210, sell 28 units at price 25996.600336, investment 0.406624 %, total balance 111854.119561,
day 218: buy 3 units at price 2306.100036, total balance 109548.019525
day 227, sell 31 units at price 29434.500000, investment 3.174002 %, total balance 138982.519525,
day 241: buy 3 units at price 2306.100036, total balance 136676.419489
day 251, sell 34 units at price 34867.000000, investment 3.292675 %, total balance 171543.419489,

total gained 170543.419489, total investment 17054.341949 %
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
