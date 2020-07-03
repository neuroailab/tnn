# Temporal Neural Networks

Run models in time.

# Installation

```
git clone https:/github.com/neuroailab/tnn
pip install -e tnn
```

*(`-e` installs a developer version such that you can always update your code to the latest)*

# Usage

Look at `tutorials`. `tutorials/alexnet_example.py` demonstrates the basic unrolling API with AlexNet. `tutorials/customcell_example.py` shows how to pass a custom cell to a model, and add edges. 

`tnn/convrnn.py` contains examples of standard ConvRNN cells in the literature. `tnn/resnetrnn.py` contains the Reciprocal Gated Cell implementation (see https://arxiv.org/abs/1807.00053 for details). 

`json` contains a set of example graphs including 5 layer LSTM and Reciprocal Gated models. To use them with the `customcell_example.py`, set the global variables `MODEL_JSON = 5L_imnet128_lstm345` and `CUSTOM_CELL = tnn_ConvLSTMCell`. You will also need to set the INPUT_LAYER and READOUT_LAYER to match the model JSON.

# Contributors

- Jonas Kubilius (MIT)
- Daniel L.K. Yamins (Stanford)
- Maryann Rui (Berkeley)
- Harry Bleyan (MIT)
- Aran Nayebi (Stanford)
- Daniel Bear (Stanford)

# License

MIT
