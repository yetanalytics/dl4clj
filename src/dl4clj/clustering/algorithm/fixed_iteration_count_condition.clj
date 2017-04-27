(ns dl4clj.clustering.algorithm.fixed-iteration-count-condition
  (:import [org.deeplearning4j.clustering.algorithm.condition FixedIterationCountCondition])
  (:require [dl4clj.clustering.algorithm.clustering-algorithm-condition :refer [is-satisfied?]]))

(defn iteration-count-greater-than
  [iteration-count]
  (.iterationCountGreaterThan iteration-count))
