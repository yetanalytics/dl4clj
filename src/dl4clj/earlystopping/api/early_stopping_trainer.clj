(ns ^{:doc "Interface for early stopping trainers.

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/trainer/IEarlyStoppingTrainer.html"}
    dl4clj.earlystopping.api.early-stopping-trainer
  (:import [org.deeplearning4j.earlystopping.trainer IEarlyStoppingTrainer]
           [org.deeplearning4j.earlystopping EarlyStoppingResult]
           [org.deeplearning4j.spark.earlystopping SparkEarlyStoppingTrainer])
  (:require [clojure.core.match :refer [match]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; local
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn fit-trainer!
  "Conduct early stopping training

  returns an early stopping result"
  [trainer]
  (match [trainer]
         [(_ :guard seq?)]
         `(.fit ~trainer)
         :else
         (.fit trainer)))

(defn set-trainer-listeners!
  "Set the early stopping listener

  returns the trainer

  :listener (listener), a listener object that implements the
  early-stopping-listener interface
  - see : TBD"
  [& {:keys [trainer listener]
      :as opts}]
  (match [opts]
         [{:trainer (_ :guard seq?)
           :listener (_ :guard seq?)}]
         `(doto ~trainer (.setListener ~listener))
         :else
         (doto trainer (.setListener listener))))

(defn get-best-model-from-result
  "returns the model within the early stopping result"
  [early-stopping-result]
  (match [early-stopping-result]
         [(_ :guard seq?)]
         `(.getBestModel ~early-stopping-result)
         :else
         (.getBestModel early-stopping-result)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; spark
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn fit-spark-es-trainer!
  "Fit the network used to create the early stopping trainer
   with the data supplied.

  :es-trainer (early stopping trainer), the object created by new-spark-early-stopping-trainer

  :rdd (JavaRdd <dataset or Multidataset>) the data to train on

  :multi-ds? (boolean), is the dataset within the RDD a multi-data-set?
   defaults to false, so expects rdd to contain a DataSet"
  [& {:keys [es-trainer rdd multi-ds?]
      :or {multi-ds? false}
      :as opts}]
  (match [opts]
         [{:es-trainer (_ :guard seq?)
           :rdd (_ :guard seq?)
           :multi-ds? true}]
         `(doto ~es-trainer (.fitMulti ~rdd))
         [{:es-trainer (_ :guard seq?)
           :rdd (_ :guard seq?)
           :multi-ds? (:or false nil)}]
         `(doto ~es-trainer (.fit ~rdd))
         [{:es-trainer _
           :rdd _
           :multi-ds? true}]
         (doto es-trainer (.fitMulti rdd))
         [{:es-trainer _
           :rdd _
           :multi-ds? (:or false nil)}]
         (doto es-trainer (.fit rdd))))

(defn get-score-spark-es-trainer
  "returns the score of the model trained via spark"
  [fit-es-trainer]
  (match [fit-es-trainer]
         [(_ :guard seq?)]
         `(.getScore ~fit-es-trainer)
         :else
         (.getScore fit-es-trainer)))
