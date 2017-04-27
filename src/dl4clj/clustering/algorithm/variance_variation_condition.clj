(ns dl4clj.clustering.algorithm.variance-variation-condition
  (:import [org.deeplearning4j.clustering.algorithm.condition VarianceVariationCondition])
  (:require [dl4clj.clustering.algorithm.clustering-algorithm-condition :refer [is-satisfied?]]))

(defn variance-variation-less-than
  [variance-variation period]
  (.varianceVariationLessThan variance-variation period))
