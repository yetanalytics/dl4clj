(ns ^{:doc "Implementation of the methods found in the RecurrentLayer Interface.
fns are for recurrent layers (LSTM, RecurrentLayer)
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/layers/RecurrentLayer.html"}
    dl4clj.nn.api.layers.recurrent-layer
  (:import [org.deeplearning4j.nn.api.layers RecurrentLayer]))

(defn rnn-activate-using-stored-state
  "Similar to rnnTimeStep, this method is used for activations using the state
  stored in the stateMap as the initialization.

  input (INDArray) of input values to the layer
  store-last-for-tbptt? (boolean), save state to be used in tbptt
  training? (boolean) is the model currently in training or not?"
  [& {:keys [rnn-layer input training? store-last-for-tbptt?]}]
  (.rnnActivateUsingStoredState rnn-layer input training? store-last-for-tbptt?))

(defn rnn-clear-previous-state!
  "Reset/clear the stateMap for rnn-time-step and
  tBpttStateMap for rnn-activate-using-stored-state

  returns the rnn-layer"
  [rnn-layer]
  (doto rnn-layer
    (.rnnClearPreviousState)))

(defn rnn-get-prev-state
  "Returns a shallow copy of the RNN stateMap (that contains the stored history
  for use in fns such as rnn-time-step"
  [rnn-layer]
  (.rnnGetPreviousState rnn-layer))

(defn rnn-get-tbptt-state
  "Get the RNN truncated backpropagations through time (TBPTT) state for the recurrent layer."
  [rnn-layer]
  (.rnnGetTBPTTState rnn-layer))

(defn rnn-set-tbptt-state!
  "Set the RNN truncated backpropagations through time (TBPTT) state for the recurrent layer.
  and returns the layer

  state is a map of {string indArray}"
  [& {:keys [rnn-layer state]}]
  (doto rnn-layer
    (.rnnSetTBPTTState state)))

(defn rnn-set-prev-state!
  "Set the stateMap (stored history) and return the layer.

  state is a map of {string indArray}"
  [& {:keys [rnn-layer state]}]
  (doto rnn-layer
    (.rnnSetPreviousState state)))

(defn rnn-time-step
  "Do one or more time steps using the previous time step state stored in stateMap.
  Can be used to efficiently do forward pass one or n-steps at a time (instead of doing forward pass always from t=0)
  If stateMap is empty, default initialization (usually zeros) is used
  Implementations also update stateMap at the end of this method

  input should be an INDArray of input values to the layer"
  [& {:keys [rnn-layer input]}]
  (.rnnTimeStep rnn-layer input))

(defn tbptt-backprop-gradient
  "Returns the Truncated BPTT gradient

  epsilon should be an INDArray
  tbptt-back-length is an integer"
  [& {:keys [rnn-layer epsilon tbptt-back-length]}]
  (.tbpttBackpropGradient rnn-layer epsilon tbptt-back-length))
