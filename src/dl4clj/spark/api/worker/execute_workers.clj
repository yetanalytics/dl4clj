(ns ^{:doc "The fns here should not need to be called by the user.  Happens behind the scene
 see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/api/worker/package-summary.html"}
    dl4clj.spark.api.worker.execute-workers
  (:import [org.deeplearning4j.spark.api.worker
            ExecuteWorkerMultiDataSetFlatMap
            ExecuteWorkerPathMDSFlatMap
            ExecuteWorkerPDSMDSFlatMap
            ExecuteWorkerPathFlatMap
            ExecuteWorkerPDSFlatMap
            ExecuteWorkerFlatMap])
  (:require [dl4clj.utils :refer [generic-dispatching-fn]]))

;; dont think this is a core ns
;; will be removed in the core branch

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method constructor calling
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti execute-workers generic-dispatching-fn)

(defmethod execute-workers :flat-map [opts]
  (let [worker (:worker (:flat-map opts))]
    (ExecuteWorkerFlatMap. worker)))

(defmethod execute-workers :pds-flat-map [opts]
  (let [worker (:worker (:pds-flat-map opts))]
    (ExecuteWorkerPDSFlatMap. worker)))

(defmethod execute-workers :path-flat-map [opts]
  (let [worker (:worker (:path-flat-map opts))]
    (ExecuteWorkerPathFlatMap. worker)))

(defmethod execute-workers :pdsmds-flat-map [opts]
  (let [worker (:worker (:pdsmds-flat-map opts))]
    (ExecuteWorkerPDSMDSFlatMap. worker)))

(defmethod execute-workers :path-mds-flat-map [opts]
  (let [worker (:worker (:path-mds-flat-map opts))]
    (ExecuteWorkerPathMDSFlatMap. worker)))

(defmethod execute-workers :multi-ds-flat-map [opts]
  (let [worker (:worker (:multi-ds-flat-map opts))]
    (ExecuteWorkerMultiDataSetFlatMap. worker)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns with doc strings
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-execute-worker-flat-map
  "A FlatMapFunction for executing training on DataSets."
  [worker]
  (execute-workers {:flat-map {:worker worker}}))

(defn new-execute-worker-multi-ds-flat-map
  "A FlatMapFunction for executing training on MultiDataSets."
  [worker]
  (execute-workers {:multi-ds-flat-map {:worker worker}}))

(defn new-execute-worker-path-flat-map
  "A FlatMapFunction for executing training on serialized DataSet objects,
  that can be loaded from a path (local or HDFS)

  path is specified in the worker passed in"
  [worker]
  (execute-workers {:path-flat-map {:worker worker}}))

(defn new-execute-worker-path-mds-flat-map
  "A FlatMapFunction for executing training on serialized DataSet objects,
  that can be loaded from a path (local or HDFS)

  path is specified in the worker passed in

  -- need to look into what makes this different from the above fn"
  [worker]
  (execute-workers {:path-mds-flat-map {:worker worker}}))

(defn new-execute-worker-pds-flat-map
  "A FlatMapFunction for executing training on serialized DataSet objects,
  that can be loaded using a PortableDataStream"
  [worker]
  (execute-workers {:pds-flat-map {:worker worker}}))

(defn new-execute-worker-pdsmds-flat-map
  "A FlatMapFunction for executing training on serialized MultiDataSet objects,
  that can be loaded using a PortableDataStream"
  [worker]
  (execute-workers {:pdsmds-flat-map {:worker worker}}))
