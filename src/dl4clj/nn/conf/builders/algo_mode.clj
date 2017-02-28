(ns dl4clj.nn.conf.builders.algo-mode
  (:require [clojure.string :as s])
  (:import [org.deeplearning4j.nn.conf.layers ConvolutionLayer$AlgoMode]))

(defn value-of [k]
  (if (string? k)
    (ConvolutionLayer$AlgoMode/valueOf k)
    (ConvolutionLayer$AlgoMode/valueOf (s/replace (s/upper-case (name k)) "-" "_"))))

(defn values []
  (map #(keyword (s/replace (s/lower-case (str %)) "_" "-")) (ConvolutionLayer$AlgoMode/values)))

(comment

  (map value-of (values))
  (value-of :no-workspace)
  (value-of :prefer-fastest)
  )
