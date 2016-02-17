(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/Distribution"}
  dl4clj.nn.conf.distribution.distribution
  (:import [org.deeplearning4j.nn.conf.distribution Distribution]))

(defmulti distribution (fn [opts] (first (keys opts))))


