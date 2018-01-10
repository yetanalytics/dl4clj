(ns dl4clj.nn.api.input-pre-processors
  (:import [org.deeplearning4j.nn.conf InputPreProcessor])
  (:require [dl4clj.constants :as constants]
            [dl4clj.utils :refer [obj-or-code? eval-if-code]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [clojure.core.match :refer [match]]))

(defn backprop
  "Reverse the preProcess during backprop."
  [& {:keys [pp output mini-batch-size as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pp (_ :guard seq?)
           :output (:or (_ :guard vector?)
                        (_ :guard seq?))
           :mini-batch-size (:or (_ :guard seq?)
                                 (_ :guard number?))}]
         (obj-or-code?
          as-code?
          `(.backprop ~pp (vec-or-matrix->indarray ~output) (int ~mini-batch-size)))
         :else
         (let [[p o-vec mbs-n] (eval-if-code [pp seq?]
                                             [output seq?]
                                             [mini-batch-size seq? number?])]
           (.backprop p (vec-or-matrix->indarray o-vec) mbs-n))))

(defn pre-process
  "Pre preProcess input/activations for a multi layer network"
  [& {:keys [pp input mini-batch-size as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pp (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :mini-batch-size (:or (_ :guard seq?)
                                 (_ :guard number?))}]
         (obj-or-code?
          as-code?
          `(.preProcess ~pp (vec-or-matrix->indarray ~input) (int ~mini-batch-size)))
         :else
         (let [[p i-vec mbs-n] (eval-if-code [pp seq?]
                                             [input seq?]
                                             [mini-batch-size seq? number?])]
           (.preProcess p (vec-or-matrix->indarray i-vec) mbs-n))))

(defn feed-forward-mask-array
  [& {:keys [pp mask-array current-mask-state mini-batch-size as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pp (_ :guard seq?)
           :mask-array (:or (_ :guard vector?)
                            (_ :guard seq?))
           :current-mask-state (:or (_ :guard keyword?)
                                    (_ :guard seq?))
           :mini-batch-size (:or (_ :guard seq?)
                                 (_ :guard number?))}]
         (obj-or-code?
          as-code?
          `(.feedForwardMaskArray
           ~pp (vec-or-matrix->indarray ~mask-array) (constants/value-of {:mask-state ~current-mask-state})
           ~mini-batch-size))
         :else
         (let [[p m-vec state mbs-n] (eval-if-code [pp seq?]
                                                   [mask-array seq?]
                                                   [current-mask-state seq? keyword?]
                                                   [mini-batch-size seq? number?])]
          (.feedForwardMaskArray
          p (vec-or-matrix->indarray m-vec) (constants/value-of {:mask-state state})
          mbs-n))))

(defn get-output-type
  "For a given type of input to this preprocessor, what is the type of the output?

  :input-type (map), the input to the cnn layer
  - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants"
  [& {:keys [pp input-type as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:pp (_ :guard seq?)
           :input-type (:or (_ :guard map?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getOutputType ~pp (constants/input-types ~input-type)))
         :else
         (let [[p i-type] (eval-if-code [pp seq?]
                                        [input-type seq? map?])]
           (.getOutputType p (constants/input-types i-type)))))
