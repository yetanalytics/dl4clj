(ns dl4clj.nn.conf.distribution.normal
  (:import [org.deeplearning4j.nn.conf.distribution NormalDistribution]))

(defn get-mean [this]
  (.getMean this))

(defn get-std [this]
  (.getStd this))

(defn set-mean [this m]
  (.setMean this (double m)))

(defn set-std [this std]
  (.setStd this (double std)))
