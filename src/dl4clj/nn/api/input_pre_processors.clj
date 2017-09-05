(ns dl4clj.nn.api.input-pre-processors
  (:import [org.deeplearning4j.nn.conf InputPreProcessor])
  (:require [dl4clj.constants :as constants]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [clojure.core.match :refer [match]]))

(defn backprop
  "Reverse the preProcess during backprop."
  [& {:keys [pp output mini-batch-size]
      :as opts}]
  (match [opts]
         [{:pp (_ :guard seq?)
           :output (:or (_ :guard vector?)
                        (_ :guard seq?))
           :mini-batch-size (:or (_ :guard seq?)
                                 (_ :guard number?))}]
         `(.backprop ~pp (vec-or-matrix->indarray ~output) (int ~mini-batch-size))
         :else
         (.backprop pp (vec-or-matrix->indarray output) mini-batch-size)))

(defn pre-process
  "Pre preProcess input/activations for a multi layer network"
  [& {:keys [pp input mini-batch-size]
      :as opts}]
  (match [opts]
         [{:pp (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :mini-batch-size (:or (_ :guard seq?)
                                 (_ :guard number?))}]
         `(.preProcess ~pp (vec-or-matrix->indarray ~input) (int ~mini-batch-size))
         :else
         (.preProcess pp (vec-or-matrix->indarray input) mini-batch-size)))

(defn feed-forward-mask-array
  [& {:keys [pp mask-array current-mask-state mini-batch-size]
      :as opts}]
  (match [opts]
         [{:pp (_ :guard seq?)
           :mask-array (:or (_ :guard vector?)
                            (_ :guard seq?))
           :current-mask-state (:or (_ :guard keyword?)
                                    (_ :guard seq?))
           :mini-batch-size (:or (_ :guard seq?)
                                 (_ :guard number?))}]
         `(.feedForwardMaskArray
           ~pp (vec-or-matrix->indarray ~mask-array) (constants/value-of {:mask-state ~current-mask-state})
           ~mini-batch-size)
         :else
         (.feedForwardMaskArray
          pp (vec-or-matrix->indarray mask-array) (constants/value-of {:mask-state current-mask-state})
          mini-batch-size)))

(defn get-output-type
  "For a given type of input to this preprocessor, what is the type of the output?

  :input-type (map), the input to the cnn layer
  - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants"
  [& {:keys [pp input-type]
      :as opts}]
  (match [opts]
         [{:pp (_ :guard seq?)
           :input-type (:or (_ :guard map?)
                            (_ :guard seq?))}]
         `(.getOutputType ~pp (constants/input-types ~input-type))
         :else
         (.getOutputType pp (constants/input-types input-type))))
