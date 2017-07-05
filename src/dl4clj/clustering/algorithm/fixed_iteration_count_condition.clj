(ns dl4clj.clustering.algorithm.fixed-iteration-count-condition
  (:import [org.deeplearning4j.clustering.algorithm.condition FixedIterationCountCondition]))

(defn iteration-count-greater-than
  [iteration-count]
  (.iterationCountGreaterThan iteration-count))
