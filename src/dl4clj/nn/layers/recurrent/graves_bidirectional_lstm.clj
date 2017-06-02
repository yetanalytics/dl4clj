(ns ^{:doc "Based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
Implementation of the class GravesBidirectionalLSTM in dl4j
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/recurrent/GravesBidirectionalLSTM.html"}
    dl4clj.nn.layers.recurrent.graves-bidirectional-lstm
  (:import [org.deeplearning4j.nn.layers.recurrent GravesBidirectionalLSTM])
  #_(:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.api.layers.recurrent-layer :refer :all]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]))
(defn new-bidirectional-lstm-layer
  "creates a new bidirectional lstm layer given a neural net conf and
  optionally some input data"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (GravesBidirectionalLSTM. conf input)
    (GravesBidirectionalLSTM. conf)))
