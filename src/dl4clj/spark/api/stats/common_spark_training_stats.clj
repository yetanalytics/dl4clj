(ns dl4clj.spark.api.stats.common-spark-training-stats
  (:import [org.deeplearning4j.spark.api.stats
            CommonSparkTrainingStats
            CommonSparkTrainingStats$Builder]))

(defn new-common-stats-builder
  "A SparkTrainingStats implementation for common stats functionality used by most workers
   - not a user facing fn

   :builder (stats-builder), a way to provide an existing unbuilt common stats builder
    - if not supplied, defaults to a fresh CommonSparkTrainingStats builder

   :build? (boolean), a way to specify if you want the builder to bo built or not
    - defaults to true

   :master-specific-stats (SparkTrainingStats), stats objects specific to a cetrain master
    - not yet implemented, but classes that implement dl4clj.spark.api.interfaces.training-stats

   :worker-get-time-ms (list), list of event stats
    - classes that implement the EventStats interface (not yet implemented)
    - for now, see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/stats/EventStats.html

   :worker-get-initial-model-time (list), list of event stats
    - classes that implement the EventStats interface (not yet implemented)
    - for now, see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/stats/EventStats.html

   :worker-process-mini-batch-time (list), list of event stats
    - classes that implement the EventStats interface (not yet implemented)
    - for now, see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/stats/EventStats.html

   :worker-total-time (list), list of event stats
    - classes that implement the EventStats interface (not yet implemented)
    - for now, see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/stats/EventStats.html"
  [& {:keys [builder build? master-specific-stats worker-get-time-ms
             worker-get-inital-model-time worker-process-mini-batch-time
             worker-total-time]
      :or {builder (CommonSparkTrainingStats$Builder.)
           build? true}
      :as opts}]
  (cond-> builder
    (contains? opts :master-specific-stats)
    (.trainingMasterSpecificStats master-specific-stats)
    (contains? opts :worker-get-time-ms)
    (.workerFlatMapDataSetGetTimesMs worker-get-time-ms)
    (contains? opts :worker-get-inital-model-time)
    (.workerFlatMapGetInitialModelTimeMs worker-get-inital-model-time)
    (contains? opts :worker-process-mini-batch-time)
    (.workerFlatMapProcessMiniBatchTimesMs worker-process-mini-batch-time)
    (contains? opts :worker-total-time)
    (.workerFlatMapTotalTimeMs worker-total-time)
    (true? build?) .build))
