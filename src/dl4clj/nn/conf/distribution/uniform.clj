(ns dl4clj.nn.conf.distribution.uniform
  (:import [org.deeplearning4j.nn.conf.distribution UniformDistribution]))

(defn get-lower [this]
  (.getLower this))

(defn get-upper [this]
  (.getUpper this))

(defn set-lower [this l]
  (.setLower this (double l)))

(defn set-upper [this u]
  (.setUpper this (double u)))
