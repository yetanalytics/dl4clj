;;deprecated
(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/UniformDistribution.html"}
  dl4clj.nn.conf.distribution.uniform-distribution
  (:require [dl4clj.nn.conf.distribution.distribution :refer [distribution]])
  (:import [org.deeplearning4j.nn.conf.distribution UniformDistribution]))

(defn uniform-distribution [lower upper]
  (UniformDistribution. lower upper))

(defmethod distribution :uniform [opt]
  (uniform-distribution (:lower (:uniform opt)) (:upper (:uniform opt))))


(comment

  (uniform-distribution 0.3 10)
  (distribution {:uniform {:lower -0.01, :upper 0.01}})

  )
