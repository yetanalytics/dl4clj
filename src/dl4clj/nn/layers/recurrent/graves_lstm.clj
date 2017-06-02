(ns ^{:doc "Based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
Implementation of the GravesLSTM class in dl4j
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/recurrent/GravesLSTM.html"}
    dl4clj.nn.layers.recurrent.graves-lstm
  (:import [org.deeplearning4j.nn.layers.recurrent GravesLSTM])
  #_(:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.api.layers.recurrent-layer :refer :all]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 merge-layers!
                                                 validate-input!]]))

(defn new-graves-lstm-layer
  "creates a new LSTM layer given a neural net conf and
  optionally some input data"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (GravesLSTM. conf input)
    (GravesLSTM. conf)))
