(ns dl4clj.nn.conf.builders.convolution-type
  (:require [clojure.string :as s])
  (:import [org.nd4j.linalg.convolution Convolution$Type]
           [org.deeplearning4j.nn.conf.layers ConvolutionLayer$AlgoMode]))

(defn value-of [k]
  (if (string? k)
    (Convolution$Type/valueOf k)
    (Convolution$Type/valueOf (s/replace (s/upper-case (name k)) "-" "_"))))

(defn values []
  (map #(keyword (s/replace (s/lower-case (str %)) "_" "-")) (Convolution$Type/values)))

(comment

  (map value-of (values))
  (value-of :full)
  (value-of :same)
  (value-of :valid)

  )
