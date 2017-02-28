;;deprecated
(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/distribution/NormalDistribution.html"}
  dl4clj.nn.conf.distribution.normal-distribution
  (:require [dl4clj.nn.conf.distribution.distribution :refer [distribution]])
  (:import [org.deeplearning4j.nn.conf.distribution NormalDistribution]))


(defn normal-distribution [mean std]
  (NormalDistribution. mean std))

(defmethod distribution :normal [opt]
  (normal-distribution (:mean (:normal opt)) (:std (:normal opt))))

(comment

  (normal-distribution 10 0.3)
  (distribution {:normal {:mean 0.0, :std 0.3}})


)
