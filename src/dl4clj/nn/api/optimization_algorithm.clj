(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/OptimizationAlgorithm.html"}
  dl4clj.nn.api.optimization-algorithm
  (:import [org.deeplearning4j.nn.api OptimizationAlgorithm]))

(defn value-of [k]
  (if (string? k)
    (OptimizationAlgorithm/valueOf k)
    (OptimizationAlgorithm/valueOf (clojure.string/replace (clojure.string/upper-case (name k)) "-" "_"))))

(defn values []
  (map #(keyword (clojure.string/replace (clojure.string/lower-case (str %)) "_" "-")) (OptimizationAlgorithm/values)))

(comment

  (values)
  (value-of :stochastic-gradient-descent)
  (value-of "STOCHASTIC_GRADIENT_DESCENT")


)
