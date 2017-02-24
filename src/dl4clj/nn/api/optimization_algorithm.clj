(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/OptimizationAlgorithm.html"}
    dl4clj.nn.api.optimization-algorithm
  (:require [clojure.string :as s])
  (:import [org.deeplearning4j.nn.api OptimizationAlgorithm]))

(defn value-of [k]
  (if (string? k)
    (OptimizationAlgorithm/valueOf k)
    (OptimizationAlgorithm/valueOf (s/replace (s/upper-case (name k)) "-" "_"))))

(defn values []
  (map #(keyword (s/replace (s/lower-case (str %)) "_" "-")) (OptimizationAlgorithm/values)))

(comment

  (map value-of (values))
  (value-of :line-gradient-descent)
  (value-of :conjugate-gradient)
  (value-of :hessian-free)
  (value-of :lbfgs)
  (value-of :stochastic-gradient-descent)
  (value-of "STOCHASTIC_GRADIENT_DESCENT")

)
