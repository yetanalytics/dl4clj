(ns ^{:doc "SparkTrainingStats is an interface that is used for accessing training statistics, for multiple TrainingMaster implementations.

The idea is that for debugging purposes, we want to collect a number of statistics related to the training. However, these statistics will vary, depending on which the type of training we are doing. Specifically, both the keys (number/names of stats) and their actual values (types/classes) can vary.

The interface here operates essentially as a Map<String,Object>. Note however that SparkTrainingStats instances may be nested: for example a ParameterAveragingTrainingMasterStats may have a CommonSparkTrainingStats instance which may in turn have a ParameterAveragingTrainingWorkerStats instance.

see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/api/stats/SparkTrainingStats.html"}
    dl4clj.spark.api.training-stats
  (:import [org.deeplearning4j.spark.api.stats SparkTrainingStats])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code? eval-if-code]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; getters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-key-set-for-stats
  "returns the key set for the training stats instance"
  [training-stats & {:keys [as-code?]
                     :or {as-code? true}}]
  (match [training-stats]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getKeySet ~training-stats))
         :else
         (.getKeySet training-stats)))

(defn get-nested-training-stats
  "Return the nested training stats - if any."
  [training-stats & {:keys [as-code?]
                     :or {as-code? true}}]
  (match [training-stats]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getNestedTrainingStats ~training-stats))
         :else
         (.getNestedTrainingStats training-stats)))

(defn get-short-name-for-key
  "Return a short (display) name for the given key.

  :stat-key (str), the identifier for the key you want the short name of"
  [& {:keys [training-stats stat-key as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:training-stats (_ :guard seq?)
           :stat-key (:or (_ :guard string?)
                          (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getShortNameForKey ~training-stats ~stat-key))
         :else
         (let [[s k] (eval-if-code [training-stats seq?]
                                   [stat-key seq? string?])]
          (.getShortNameForKey s k))))

(defn get-value-for-key
  "Get the statistic value for this key"
  [& {:keys [training-stats stat-key as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:training-stats (_ :guard seq?)
           :stat-key (:or (_ :guard string?)
                          (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getValue ~training-stats ~stat-key))
         :else
         (let [[t s] (eval-if-code [training-stats seq?] [stat-key seq? string?])]
           (.getValue t s))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn add-other-training-stats!
  "Combine the two training stats instances.

  :other-training-stats (SparkTrainingStats), training stats tracker to add to the first one
   - see: TBD

  returns the first instance with the other added in"
  [& {:keys [training-stats other-training-stats as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:training-stats (_ :guard seq?)
           :other-training-stats (_ :guard seq?)}]
         (obj-or-code?
          as-code?
          `(doto ~training-stats (.addOtherTrainingStats ~other-training-stats)))
         :else
         (let [[s1 s2] (eval-if-code [training-stats seq?]
                                     [other-training-stats seq?])]
           (doto s1 (.addOtherTrainingStats s2)))))

(defn default-include-in-plots
  "When plotting statistics, we don't necessarily want to plot everything.

  :to-include (str), the key to include in the stat plot
   - need to look into possible options for to-include

  returns the training-stats instance"
  [& {:keys [training-stats stat-key-to-include as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:training-stats (_ :guard seq?)
           :stat-key-to-include (:or (_ :guard string?)
                                     (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~training-stats (.defaultIncludeInPlots ~stat-key-to-include)))
         :else
         (let [[t k] (eval-if-code [training-stats seq?]
                                   [stat-key-to-include seq? string?])]
           (doto t (.defaultIncludeInPlots k)))))

(defn export-stat-files!
  "Export the stats as a collection of files.

  :path (str), the file path to export the files to

  :spark-context (sc), the spark context used in training
   - of type org.apache.spark.SparkContext
   - creation of these has not been implemented yet

  returns the training-stats instance"
  [& {:keys [training-stats path spark-context as-code?]
      :or {as-code? true}
      :as opts}]
  (let [[t p c] (eval-if-code [training-stats seq?]
                              [path seq? string?]
                              [spark-context seq?])]
   (doto t (.exportStatFiles p c))))

(defn stats-as-string
  "get the stats as a string"
  [training-stats & {:keys [as-code?]
                     :or {as-code? true}}]
  (match [training-stats]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.statsAsString ~training-stats))
         :else
         (.statsAsString training-stats)))
