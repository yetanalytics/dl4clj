(ns ^{:doc "Weight initialization utility
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/weights/WeightInitUtil.html"}
    dl4clj.nn.weights.weight-init-util
  (:import [org.deeplearning4j.nn.weights WeightInitUtil])
  (:require [dl4clj.constants :as enum]
            [dl4clj.utils :refer [contains-many?]]))

(defn init-weights
  "Initializes a matrix with the given weight initialization scheme.

  :fan-in (double),

  :fan-out (double),

  :shape (int-array), the shape of the weight matrix

  :weight-init (keyword), one of: :distribution, :zero, :sigmoid-uniform, :uniform, :xavier, :xavier-uniform
          :xavier-fan-in, :xavier-legacy, :relu, :relu-uniform, :vi, :size
   - for more details, see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/weights/WeightInit.html

  :array-order (keyword), controls the order arrays for the weights****
   - defaults to fortran ('f'), will need to test to figure out the other ways to order the arrays

  :min-val (float), the min value of the weights

  :max-val (float), the max value of the weights

  :distribution (distribution), a distribution to sample weights from
   - see: dl4clj.nn.conf.distribution.distribution

  :param-view (INDArray), not sure the proper desc for this arg****"
  [& {:keys [fan-in fan-out shape weight-init array-order min-val max-val
             distribution param-view]
      :as opts}]
  (cond (contains-many? opts :fan-in :fan-out :shape :weight-init
                        :distribution :array-order :param-view)
        (.initWeights fan-in fan-out shape (enum/value-of {:weight-init weight-init})
                      distribution array-order param-view)
        (contains-many? opts :min-val :max-val :shape)
        (.initWeights shape min-val max-val)
        (contains-many? opts :fan-in :fan-out :shape :weight-init
                        :distribution :param-view)
        (.initWeights fan-in fan-out shape (enum/value-of {:weight-init weight-init})
                      distribution param-view)
        :else
        (assert false "you must atleast supply a matrix shape and the min/max values in that matrix")))

(defn reshape-weights
  "Reshape the parameters view, without modifying the paramsView array values.

  :shape (int array), the shape you want the weights to be converted to

  :param-view (INDArray), still not sure the best desc for this arg****

  :array-order (keyword), the order for the weights array, defaults to fortran
   - need to figure out what the other options are****"
  [& {:keys [shape param-view array-order]
      :as opts}]
  (assert (contains-many? opts :shape :param-view)
          "you must provide a matrix shape and a param-view INDArray")
  (if (contains? opts :array-order)
    (.reshapeWeights shape param-view array-order)
    (.reshapeWeights shape param-view)))
