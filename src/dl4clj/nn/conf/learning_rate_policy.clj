(ns dl4clj.nn.conf.learning-rate-policy
  (:require [dl4clj.utils :refer [camelize camel-to-dashed]])
  (:import [org.deeplearning4j.nn.conf LearningRatePolicy]))

(defn value-of [k]
  (if (string? k)
    (LearningRatePolicy/valueOf k)
    (LearningRatePolicy/valueOf (camelize (name k) true))))

(defn values []
  (map #(keyword (camel-to-dashed (.name ^LearningRatePolicy %))) (LearningRatePolicy/values)))

(comment
  (map value-of (values))
  (value-of :none)
  (value-of :exponential)
  (value-of :inverse)
  (value-of :poly)
  (value-of :sigmoid)
  (value-of :step)
  (value-of :torch-step)
  (value-of :schedule)
  (value-of :score)
  (value-of "TorchStep")
  )
