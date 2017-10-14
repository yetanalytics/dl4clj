(ns dl4clj.spark.masters.param-avg
  (:import [org.deeplearning4j.spark.impl.paramavg
            ParameterAveragingTrainingMaster
            ParameterAveragingTrainingMaster$Builder])
  (:require [dl4clj.utils :refer [contains-many? builder-fn replace-map-vals obj-or-code?]]
            [dl4clj.helpers :refer [value-of-helper]]))

(defn new-parameter-averaging-training-master
  "used for training networks on Spark.
   - for more detailed description of args, see: https://deeplearning4j.org/spark#Configuring_the_TrainingMaster

  :build? (boolean), whether or not to build the training master
   - defaults to true

  :rdd-n-examples (int), specifies how many examples are in each DataSet object
   - If you are training with pre-processed DataSet objects, this will be the size of those preprocessed DataSets
   - If you are training directly from Strings (for example, CSV data to a RDD<DataSet>) then this will usually be 1

  :n-workers (int), the number of works being used
   - (* Spark executors  number of threads per executor)
   - this should match the configuration of the cluster

  :averaging-freq (int), frequency with which to average worker params
   - too high or too low can be bad for different reasons
     - too low (such as 1), can result in a lot of network traffic
     - too high (> 20), can result in accuracy issues or problems with network convergence

  :batch-size-per-worker (int), number of examples per worker for each fit call the worker makes

  :export-dir (str), the directory which the data is exported to

  :rdd-training-approach (keyword), the approach to use when training
   - one of: :direct or :export

  :repartition-data (keyword), if/when repartitioning should be conducted for the training data
   - one of: :always, :never, or :num-partitions-workers-differs

  :repartition-strategy (keyword), defines how the repartitioning should be done
   - one of: :balanced or :spark-default

  :seed (long), Random number generator seed, used mainly for enforcing repeatable splitting on RDDs

  :save-updater? (boolean), whether the updater historical state should be saved

  :storage-level (keyword), the spark storage level for the RDD training data
   - one of: :none, :off-heap, :memory-only, :disk-only, :disk-only-2, :memory-only-2
             :memory-only-ser, :memory-only-ser-2, :memory-and-disk, :memory-and-disk-ser,
             :memory-and-disk-ser-2 or :memory-and-disk-2

  :storage-level-streams (keyword), Set the storage level RDDs used when fitting data from Streams
   - same keyword options as :storage-level
   - streams come from either PortableDataStreams (sc.binaryFiles via (fit-spark! SparkDl4jMultiLayer String))
     or string paths (fit-spark-paths! SparkDl4jMultiLayer JavaRDD)

  :training-hooks (coll), a collection of training hooks to add to the master
   - see: dl4clj.spark.api.interfaces.training-hook and dl4clj.spark.parameterserver.param-server-training-hook
    - I dont think this was set up at the time of the 0.8.0 release
    - can always make a gen-class for classes which implement dl4clj.spark.api.interfaces.training-hook

  :worker-prefetch-n-batches (int), the number of mini-batches to asynchronously
   prefetch in the worker"
  [& {:keys [build? rdd-n-examples n-workers averaging-freq
             batch-size-per-worker export-dir rdd-training-approach
             repartition-data repartition-strategy seed save-updater?
             storage-level storage-level-streams training-hooks
             worker-prefetch-n-batches as-code?]
      :or {build? true
           as-code? false}
      :as opts}]
  (let [rdd-n-e (int rdd-n-examples)
        ;; ^ spec will catch when this is no present
        b (if (contains-many? opts :rdd-n-examples :n-workers)
            `(ParameterAveragingTrainingMaster$Builder. ~rdd-n-e ~(int n-workers))
            `(ParameterAveragingTrainingMaster$Builder. ~rdd-n-e))
        method-map {:averaging-freq            '.averagingFrequency
                    :batch-size-per-worker     '.batchSizePerWorker
                    :export-dir                '.exportDirectory
                    :rdd-training-approach     '.rddTrainingApproach
                    :repartition-data          '.repartionData
                    :repartition-strategy      '.repartitionStrategy
                    :seed                      '.rngSeed
                    :save-updater?             '.saveUpdater
                    :storage-level             '.storageLevel
                    :storage-level-streams     '.storageLevelStreams
                    :training-hooks            '.trainingHooks
                    :worker-prefetch-n-batches '.workerPrefetchNumBatches}
        updated-opts {:averaging-freq           (if averaging-freq
                                                  (int averaging-freq))
                      :batch-size-per-worker    (if batch-size-per-worker
                                                  (int batch-size-per-worker))
                      :rdd-training-approach    (if rdd-training-approach
                                                  (value-of-helper
                                                   :rdd-training-approach
                                                   rdd-training-approach))
                      :repartition-data         (if repartition-data
                                                  (value-of-helper
                                                   :repartition repartition-data))
                      :repartition-strategy     (if repartition-strategy
                                                  (value-of-helper
                                                   :repartition-strategy
                                                   repartition-strategy))
                      :seed                     (if seed (long seed))
                      :storage-level            (if storage-level
                                                  (value-of-helper
                                                   :storage-level storage-level))
                      :storage-level-streams    (if storage-level-streams
                                                  (value-of-helper
                                                   :storage-level
                                                   storage-level-streams))
                      :worker-prefetch-n-batches (if worker-prefetch-n-batches
                                                   (int worker-prefetch-n-batches))}
        args (replace-map-vals (dissoc opts :build? :as-code?
                                       :rdd-n-examples :n-workers)
                               updated-opts)
        code (builder-fn b method-map args)
        add-build (if build?
                    `(.build ~code)
                    code)]
    (obj-or-code? as-code? add-build)))
