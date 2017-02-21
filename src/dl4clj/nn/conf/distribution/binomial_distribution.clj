(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/BinomialDistribution.html"}
  dl4clj.nn.conf.distribution.binomial-distribution
  (:require [dl4clj.nn.conf.distribution.distribution :refer (distribution)])
  (:import [org.deeplearning4j.nn.conf.distribution BinomialDistribution]))


(defn binomial-distribution [number-of-trials probability-of-success]
  (BinomialDistribution. number-of-trials probability-of-success))

(defmethod distribution :binomial [opt]
  (binomial-distribution (:number-of-trials (:binomial opt)) (:probability-of-success (:binomial opt))))

(comment

  (binomial-distribution 10 0.3)
  (distribution {:binomial {:number-of-trials 0, :probability-of-success 0.08}})

)
