(ns ^{:doc "Deep neural net normalization approach normalizes activations between layers brightness normalization
Implementation of the class LocalResponseNormalization in dl4j
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/normalization/LocalResponseNormalization.html"}
    dl4clj.nn.layers.normalization.local-response-normalization
  (:import [org.deeplearning4j.nn.layers.normalization LocalResponseNormalization])
  (:require [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.layers.base-layer :refer [calc-activation-mean
                                                 calc-gradient
                                                 derivative-activation
                                                 calc-error
                                                 get-input
                                                 init-params!
                                                 merge-layers!
                                                 validate-input!]]))
;;For a^i_{x,y} the activity of a neuron computed by applying kernel i at position (x,y) and applying ReLU nonlinearity, the response normalized activation b^i_{x,y} is given by:

;;x^2 = (a^j_{x,y})^2 unitScale = (k + alpha * sum_{j=max(0, i - n/2)}^{max(N-1, i + n/2)} (a^j_{x,y})^2 ) y = b^i_{x,y} = x * unitScale**-beta

;;gy = epsilon (aka deltas from previous layer) sumPart = sum(a^j_{x,y} * gb^j_{x,y}) gx = gy * unitScale**-beta - 2 * alpha * beta * sumPart/unitScale * a^i_{x,y}

(defn new-local-response-normalization-layer
  "creates a local response normalization layer given a neural net conf and
  optionally some input data"
  [& {:keys [conf input]
      :as opts}]
  (assert (contains? opts :conf) "you must supply a neural network configuration")
  (if (contains? opts :input)
    (LocalResponseNormalization. conf input)
    (LocalResponseNormalization. conf)))
