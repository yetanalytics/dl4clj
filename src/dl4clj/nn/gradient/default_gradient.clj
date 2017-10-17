(ns ^{:doc "Default gradient implementation. Basically lookup table for ndarrays
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/gradient/DefaultGradient.html"}
    dl4clj.nn.gradient.default-gradient
  (:import [org.deeplearning4j.nn.gradient DefaultGradient])
  (:require [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [dl4clj.utils :refer [obj-or-code?]]))

(defn new-default-gradient
  [& {:keys [flattened-gradient as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (if flattened-gradient
               `(DefaultGradient. (vec-or-matrix->indarray ~flattened-gradient))
               `(DefaultGradient.))]
    (obj-or-code? as-code? code)))
