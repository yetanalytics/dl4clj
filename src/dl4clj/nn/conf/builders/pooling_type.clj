(ns dl4clj.nn.conf.builders.pooling-type
  (:require [clojure.string :as s])
  (:import [org.deeplearning4j.nn.conf.layers SubsamplingLayer$PoolingType]))

(defn value-of [k]
  (if (string? k)
    (SubsamplingLayer$PoolingType/valueOf k)
    (SubsamplingLayer$PoolingType/valueOf (s/replace (s/upper-case (name k)) "-" "_"))))

(defn values []
  (map #(keyword (s/replace (s/lower-case (str %)) "_" "-")) (SubsamplingLayer$PoolingType/values)))

(comment

  (map value-of (values))
  (value-of :avg)
  (value-of :max)
  (value-of :sum)
  (value-of :pnorm)
  (value-of :none)

  )
