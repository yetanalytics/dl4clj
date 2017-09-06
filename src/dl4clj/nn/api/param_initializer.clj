(ns ^{:doc "fns from the dl4j Interface for param initializers. Param initializer for a layer
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/ParamInitializer.html"}
    dl4clj.nn.api.param-initializer
  (:import [org.deeplearning4j.nn.api ParamInitializer])
  (:require [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [clojure.core.match :refer [match]]))

;; I think this is being removed from core

(defn get-gradients-from-flattened
  "Return a map of gradients (in their standard non-flattened representation),
  taken from the flattened (row vector) :gradient-view INDArray or vec.

  :conf is a NeuralNetConfiguration"
  [& {:keys [param-init conf gradient-view]
      :as opts}]
  (.getGradientsFromFlattened param-init conf (vec-or-matrix->indarray gradient-view)))

(defn init
  "Initialize the parameters

  :conf is a NeuralNetConfiguration

  :params-view is an INDArray or vector

  :initialize-params? boolean"
  [& {:keys [param-init conf params-view initialize-params?]}]
  (.init param-init conf (vec-or-matrix->indarray params-view) initialize-params?))

(defn num-params
  "return number of params

  :conf is a NeuralNetConfiguration"
  [& {:keys [param-init conf]}]
  (.numParams param-init conf))
