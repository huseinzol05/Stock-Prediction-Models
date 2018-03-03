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


## Results (not included all)

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
