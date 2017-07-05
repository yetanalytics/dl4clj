(ns dl4clj.clustering.algorithm.variance-variation-condition
  (:import [org.deeplearning4j.clustering.algorithm.condition VarianceVariationCondition]))

(defn variance-variation-less-than
  [& {:keys [variance-variation period]}]
  (.varianceVariationLessThan variance-variation period))
