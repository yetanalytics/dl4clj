(ns dl4clj.spark.training-master.param-avg
  (:import [org.deeplearning4j.spark.impl.paramavg
            ParameterAveragingTrainingMaster
            ParameterAveragingTrainingMaster$Builder])
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.constants :refer [value-of]]))

(defn new-parameter-averaging-training-master
  "used for training networks on Spark."
  [& {:keys [build? builder rdd-n-examples n-workers averaging-freq
             batch-size-per-worker export-dir rdd-training-approach
             repartition-data repartition-strategy seed save-updater?
             storage-level]
      :as opts}]
  (let [b (if (contains? opts :builder)
            builder
            (if (contains-many? opts :rdd-n-examples :n-workers)
              (ParameterAveragingTrainingMaster$Builder. rdd-n-examples n-workers)
              (ParameterAveragingTrainingMaster$Builder. rdd-n-examples)))]
    (cond-> b
      (contains? opts :averaging-freq)
      (.averagingFrequency averaging-freq)
      (contains? opts :batch-size-per-worker)
      (.batchSizePerWorker batch-size-per-worker)
      (contains? opts :export-dir)
      (.exportDirectory export-dir)
      (contains? opts :rdd-training-approach)
      (.rddTrainingApproach (value-of {:rdd-training-approach rdd-training-approach}))
      (contains? opts :repartition-data)
      (.repartionData (value-of {:repartition repartition-data}))
      (contains? opts :repartition-strategy)
      (.repartitionStrategy (value-of {:repartition-strategy repartition-strategy}))
      (contains? opts :seed)
      (.rngSeed seed)
      (contains? opts :save-updater?)
      (.saveUpdater save-updater?)
      (contains? opts :storage-level)
      (.storageLevel )














      )))
