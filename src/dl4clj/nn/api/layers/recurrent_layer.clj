(ns dl4clj.nn.api.layers.recurrent-layer
  (:import [org.deeplearning4j.nn.api.layers RecurrentLayer])
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]))

(defn rnn-activate-using-stored-state
  "Similar to rnnTimeStep, this method is used for activations using the state
  stored in the stateMap as the initialization."
  [& {:keys [this input training? store-last-for-tbptt?]}]
  (.rnnActivateUsingStoredState this input training? store-last-for-tbptt?))

(defn rnn-clear-previous-state
  "Reset/clear the stateMap for rnnTimeStep() and tBpttStateMap for rnnActivateUsingStoredState()"
  [this]
  (.rnnClearPreviousState this))

(defn rnn-get-prev-state
  "Returns a shallow copy of the RNN stateMap (that contains the stored history
  for use in methods such as rnnTimeStep"
  [this]
  (.rnnGetPreviousState this))

(defn rnn-get-tbptt-state
  "Get the RNN truncated backpropagations through time (TBPTT) state for the recurrent layer."
  [this]
  (.rnnGetTBPTTState this))

(defn rnn-set-tbptt-state
  "Set the RNN truncated backpropagations through time (TBPTT) state for the recurrent layer."
  [& {:keys [this state]}]
  (.rnnSetTBPTTState this state))

(defn rnn-set-prev-state
  "Set the stateMap (stored history)."
  [& {:keys [this state]}]
  (.rnnSetPreviousState this state))

(defn rnn-time-step
  "Do one or more time steps using the previous time step state stored in stateMap.
  Can be used to efficiently do forward pass one or n-steps at a time (instead of doing forward pass always from t=0)
  If stateMap is empty, default initialization (usually zeros) is used
  Implementations also update stateMap at the end of this method"
  [& {:keys [this input]}]
  (.rnnTimeStep this input))

(defn tbptt-backprop-gradient
  "Truncated BPTT equivalent of Layer.backpropGradient()."
  [& {:keys [this epsilon tbptt-back-length]}]
  (.tbpttBackpropGradient this epsilon tbptt-back-length))
