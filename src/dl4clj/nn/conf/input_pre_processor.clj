(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/InputPreProcessor.html"}
  dl4clj.nn.conf.input-pre-processor
  (:import [org.deeplearning4j.nn.conf InputPreProcessor]))


(defn backprop
  "Reverse the preProcess during backprop."
  [^InputPreProcessor this output mini-batch-size]
  (.backprop this output (int mini-batch-size)))

(defn clone
  [^InputPreProcessor this]
  (.clone this))

(defn pre-process
  "Pre preProcess input/activations for a multi layer network"
  [^InputPreProcessor this input mini-batch-size]
  (.preProcess this input (int mini-batch-size)))

