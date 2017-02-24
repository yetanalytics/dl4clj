(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/weights/WeightInit.html"}
  dl4clj.nn.weights.weight-init
  (:import [org.deeplearning4j.nn.weights WeightInit])
  (:require [clojure.string :as s]))

(defn value-of [k]
  (if (string? k)
    (WeightInit/valueOf k)
    (WeightInit/valueOf (s/replace (s/upper-case (name k)) "-" "_"))))

(defn values []
  (map #(keyword (s/lower-case %)) (WeightInit/values)))

(comment

  (map value-of (values))
  (value-of :distribution)
  (value-of :zero)
  (value-of :sigmoid-uniform)
  (value-of :uniform)
  (value-of :xavier)
  (value-of :xavier-uniform)
  (value-of :xavier-fan-in)
  (value-of :xavier-legacy)
  (value-of :relu)
  (value-of :relu-uniform)
  (value-of :vi)
  (value-of :size)
  (value-of :normalized)
  (value-of "NORMALIZED")

)
