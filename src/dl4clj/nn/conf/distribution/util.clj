(ns dl4clj.nn.conf.distribution.util
  (:import [org.deeplearning4j.nn.conf.distribution
            BinomialDistribution
            UniformDistribution
            NormalDistribution]))

(defn equals? [this obj]
  (.equals this obj))

(defn hash-code [this]
  (.hashCode this))

(defn to-string [this]
  (.toString this))
