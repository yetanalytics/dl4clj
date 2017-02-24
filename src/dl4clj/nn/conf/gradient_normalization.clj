(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/GradientNormalization.html"}
  dl4clj.nn.conf.gradient-normalization
  (:require [dl4clj.utils :refer (camelize camel-to-dashed)])
  (:import [org.deeplearning4j.nn.conf GradientNormalization]))

(defn value-of [k]
  (if (string? k)
    (GradientNormalization/valueOf k)
    (GradientNormalization/valueOf (camelize (name k) true))))

(defn values []
  (map #(keyword (camel-to-dashed (.name ^GradientNormalization %))) (GradientNormalization/values)))


(comment

  (map value-of (values))
  (value-of :none)
  (value-of :renormalize-l2-per-layer)
  (value-of :renormalize-l2-per-param-type)
  (value-of :clip-element-wise-absolute-value)
  (value-of :clip-l2-per-layer)
  (value-of :clip-l2-per-param-type)
  (value-of "RenormalizeL2PerLayer")
)
