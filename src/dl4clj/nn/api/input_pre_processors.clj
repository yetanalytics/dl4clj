(ns dl4clj.nn.api.input-pre-processors
  (:import [org.deeplearning4j.nn.conf InputPreProcessor])
  (:require [dl4clj.constants :as constants]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn backprop
  "Reverse the preProcess during backprop."
  [& {:keys [pp output mini-batch-size]}]
  (.backprop pp (vec-or-matrix->indarray output) mini-batch-size))

(defn pre-process
  "Pre preProcess input/activations for a multi layer network"
  [& {:keys [pp input mini-batch-size]}]
  (.preProcess pp (vec-or-matrix->indarray input) mini-batch-size))

(defn feed-forward-mask-array
  [& {:keys [pp mask-array current-mask-state mini-batch-size]}]
  (.feedForwardMaskArray
   pp (vec-or-matrix->indarray mask-array) (constants/value-of {:mask-state current-mask-state})
   mini-batch-size))

(defn get-output-type
  "For a given type of input to this preprocessor, what is the type of the output?

  :input-type (map), the input to the cnn layer
  - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants"
  [& {:keys [pp input-type]}]
  (.getOutputType pp (constants/input-types input-type)))
