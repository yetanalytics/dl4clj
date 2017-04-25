(ns ^{:doc "Implementation of the methods found in the IOutputLayer Interface.
fns are for output layers (those that calculate gradients with respect to a labels array)
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/layers/IOutputLayer.html"}
    dl4clj.nn.api.layers.i-output-layer
  (:import [org.deeplearning4j.nn.api.layers IOutputLayer]))

(defn compute-score
  "Compute score after labels and input have been set.

  output-layer is the output layer in question
  full-network-l1 (double) is the l1 regularization term for the model the layer is apart of
  full-network-l2 (double) is the l2 regularization term for the model the layer is aprt of
   -note: it is okay for L1 and L2 to be set to 0.0 if regularization was not used
  training? (boolean) are we traing the model or testing it?"
  [& {:keys [output-layer full-network-l1 full-network-l2 training?]}]
  (.computeScore output-layer full-network-l1 full-network-l2 training?))

(defn compute-score-for-examples
  "Compute the score for each example individually, after labels and input have been set.

  output-layer is the output layer in question
  full-network-l1 (double) is the l1 regularization term for the model the layer is apart of
  full-network-l2 (double) is the l2 regularization term for the model the layer is aprt of
   -note: it is okay for L1 and L2 to be set to 0.0 if regularization was not used"
  [& {:keys [output-layer full-network-l1 full-network-l2]}]
  (.computeScoreForExamples output-layer full-network-l1 full-network-l2))

(defn get-labels
  "Get the labels array previously set with set-labels!"
  [output-layer]
  (.getLabels output-layer))

(defn set-labels!
  "Set the labels array for this output layer and returns the layer

  labels is an INDArray of labels"
  [& {:keys [output-layer labels]}]
  (doto output-layer
    (.setLabels labels)))
