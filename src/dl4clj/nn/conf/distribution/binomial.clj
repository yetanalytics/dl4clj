(ns dl4clj.nn.conf.distribution.binomial
  (:import [org.deeplearning4j.nn.conf.distribution BinomialDistribution]))

(defn get-n-trials [this]
  (.getNumberOfTrials this))

(defn get-prob-of-success [this]
  (.getProbabilityOfSuccess this))

(defn set-prob-of-success [this prob]
  (.setProbabilityOfSuccess this (double prob)))
