# Sequential LSTM Model

This repository contains a Sequential LSTM-based model implemented using TensorFlow/Keras. The model is designed for time series data or sequence-based tasks, and its architecture leverages LSTM layers for capturing temporal dependencies.

## Model Architecture

The model is built using the Keras Sequential API and consists of the following layers:

1. **LSTM Layer (64 units)**: 
   - Returns sequences to allow stacking of another LSTM layer.
   - Input shape: `(timesteps, features)` where `timesteps` is `X.shape[1]` and `features` is `X.shape[2]`.
   - Dropout (20%) applied to mitigate overfitting.

2. **LSTM Layer (32 units)**:
   - Outputs the final state for downstream processing.
   - Dropout (20%) applied to mitigate overfitting.

3. **Dense Layer (1 unit)**:
   - Single neuron for output, suitable for regression tasks.

## Compilation

The model is compiled with the following configurations:
- **Optimizer**: Adam — Adaptive optimization for fast and stable convergence.
- **Loss Function**: Mean Squared Error (MSE) — Ideal for regression tasks.

```python
model.compile(optimizer='adam', loss='mse')