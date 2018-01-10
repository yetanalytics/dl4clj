(ns ^{:doc "Interface for early stopping trainers.

see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/trainer/IEarlyStoppingTrainer.html"}
    dl4clj.earlystopping.api.early-stopping-trainer
  (:import [org.deeplearning4j.earlystopping.trainer IEarlyStoppingTrainer]
           [org.deeplearning4j.earlystopping EarlyStoppingResult]
           [org.deeplearning4j.spark.earlystopping SparkEarlyStoppingTrainer])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code? eval-if-code]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; local
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn fit-trainer!
  "Conduct early stopping training

  returns an early stopping result"
  [trainer & {:keys [as-code?]
              :or {as-code? true}}]
  (match [trainer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.fit ~trainer))
         :else
         (.fit trainer)))

(defn set-trainer-listeners!
  "Set the early stopping listener

  returns the trainer

  :listener (listener), a listener object that implements the
  early-stopping-listener interface
  - see : TBD"
  [& {:keys [trainer listener as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:trainer (_ :guard seq?)
           :listener (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~trainer (.setListener ~listener)))
         :else
         (let [[trainer-obj listener-obj] (eval-if-code [trainer seq?]
                                                        [listener seq?])]
           (doto trainer-obj (.setListener listener-obj)))))

(defn get-best-model-from-result
  "returns the model within the early stopping result"
  [early-stopping-result & {:keys [as-code?]
                            :or {as-code? true}}]
  (match [early-stopping-result]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getBestModel ~early-stopping-result))
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
  [& {:keys [es-trainer rdd multi-ds? as-code?]
      :or {multi-ds? false
           as-code? true}
      :as opts}]
  (match [opts]
         [{:es-trainer (_ :guard seq?)
           :rdd (_ :guard seq?)
           :multi-ds? true}]
         (obj-or-code? as-code? `(doto ~es-trainer (.fitMulti ~rdd)))
         [{:es-trainer (_ :guard seq?)
           :rdd (_ :guard seq?)
           :multi-ds? (:or false nil)}]
         (obj-or-code? as-code? `(doto ~es-trainer (.fit ~rdd)))
         [{:es-trainer _
           :rdd _
           :multi-ds? true}]
         (let [[es-trainer-obj rdd-obj] (eval-if-code [es-trainer seq?]
                                                      [rdd seq?])]
           (doto es-trainer-obj (.fitMulti rdd-obj)))
         [{:es-trainer _
           :rdd _
           :multi-ds? (:or false nil)}]
         (let [[es-trainer-obj rdd-obj] (eval-if-code [es-trainer seq?]
                                                      [rdd seq?])]
           (doto es-trainer-obj (.fit rdd-obj)))))

(defn get-score-spark-es-trainer
  "returns the score of the model trained via spark"
  [fit-es-trainer & {:keys [as-code?]
                     :or {as-code? true}}]
  (match [fit-es-trainer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getScore ~fit-es-trainer))
         :else
         (.getScore fit-es-trainer)))
