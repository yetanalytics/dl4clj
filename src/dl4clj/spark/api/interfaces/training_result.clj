(ns ^{:doc "fn used by a type of training master, see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/api/TrainingResult.html"}
    dl4clj.spark.api.interfaces.training-result
  (:import [org.deeplearning4j.spark.api TrainingResult]))

(defn set-stats!
  "sets the stats for a training result
   - not a user facing fn

  returns a map of the supplied input after the mutation"
  [& {:keys [training-result training-stats]}]
  (.setStats training-result training-stats)
  {:param-averaging-training-result training-result
   :stats training-stats})
