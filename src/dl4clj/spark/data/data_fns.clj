(ns ^{:doc "multi method and user facing fns for the spark dataset functions

see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/data/package-summary.html"}
    dl4clj.spark.data.data-fns
  (:import [org.deeplearning4j.spark.data
            BatchAndExportMultiDataSetsFunction
            BatchAndExportDataSetsFunction
            PathToMultiDataSetFunction
            MultiDataSetExportFunction
            SplitDataSetsFunction
            PathToDataSetFunction
            DataSetExportFunction
            BatchDataSetsFunction])
  (:require [dl4clj.utils :refer [generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method for constructor calling
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti ds-fns generic-dispatching-fn)

(defmethod ds-fns :batch-and-export-ds [opts]
  (let [conf (:batch-and-export-ds opts)
        {batch-size :batch-size
         path :export-path} conf]
    (BatchAndExportDataSetsFunction. batch-size path)))

(defmethod ds-fns :batch-and-export-multi-ds [opts]
  (let [conf (:batch-and-export-ds opts)
        {batch-size :batch-size
         path :export-path} conf]
    (BatchAndExportMultiDataSetsFunction. batch-size path)))

(defmethod ds-fns :batch-ds [opts]
  (let [mbs (:batch-size (:batch-ds opts))]
    (BatchDataSetsFunction. mbs)))

(defmethod ds-fns :export-ds [opts]
  (let [p (:export-path (:export-ds opts))]
    (DataSetExportFunction. p)))

(defmethod ds-fns :export-multi-ds [opts]
  (let [p (:export-path (:export-ds opts))]
    (MultiDataSetExportFunction. p)))

(defmethod ds-fns :split-ds [opts]
  (SplitDataSetsFunction.))

(defmethod ds-fns :path-to-ds [opts]
  (PathToDataSetFunction.))

(defmethod ds-fns :path-to-multi-ds [opts]
  (PathToMultiDataSetFunction.))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-batch-and-export-ds-fn
  "Function used with (map-partition-with-idx RDD<DataSet>)
    - not yet implemented at time of writing doc string

  It does two things:
  1. Batch DataSets together, to the specified minibatch size.
     This may result in splitting or combining existing DataSet objects as required
  2. Export the DataSet objects to the specified directory.

  Naming convention for export files: (str dataset_ partition-idx jvm_uid _ idx .bin)

  :batch-size (int), the size of the batches

  :export-path (str), the directory you want to export to"
  [& {:keys [batch-size export-path]
      :as opts}]
  (ds-fns {:batch-and-export-ds opts}))

(defn new-batch-and-export-multi-ds-fn
  "Function used with (map-partition-with-idx RDD<MultiDataSet>)
   - not yet implemented at time of writing doc string

  It does two things:
  1. Batch MultiDataSets together, to the specified minibatch size.
     This may result in splitting or combining existing MultiDataSet objects as required
  2. Export the MultiDataSet objects to the specified directory.

  Naming convention for export files: (str mds_ partition-idx jvm_uid _ idx .bin)

  :batch-size (int), the size of the batches

  :export-path (str), the directory you want to export to"
  [& {:keys [batch-size export-path]
      :as opts}]
  (ds-fns {:batch-and-export-multi-ds opts}))

(defn new-batch-ds-fn
  "Function used to batch DataSet objects together.

  Typically used to combine singe-example DataSet objects out of something
  like DataVecDataSetFunction together into minibatches.

  Usage: (def single-ds-ex (map-partitions (new-batch-ds-fn n)))

  batch-size (int), the size of the batches"
  [batch-size]
  (ds-fns {:batch-ds {:batch-size batch-size}}))

(defn new-ds-export-fn
  "A function to save DataSet objects to disk/HDFS.
   - used with (for-each-partition [data JavaRDD<DataSet>] (new-ds-export-fn export-path))
     - need to implement for-each-partition at time of writing doc string

  Each DataSet object is given a random and (probably) unique name starting with
  dataset_ and ending with .bin.

  export-path (URI), the place to export to, needs to be a java.net.URI"
  [export-path]
  (ds-fns {:export-ds {:export-path export-path}}))


(defn new-multi-ds-export-fn
  "A function to save MultiDataSet objects to disk/HDFS.
   - used with (for-each-partition [data JavaRDD<MultiDataSet>] (new-ds-export-fn export-path))
     - need to implement for-each-partition at time of writing doc string

  Each MultiDataSet object is given a random and (probably) unique name starting with
  dataset_ and ending with .bin.

  export-path (URI), the place to export to, needs to be a java.net.URI"
  [export-path]
  (ds-fns {:export-multi-ds {:export-path export-path}}))

(defn new-path-to-ds-fn
  "Simple function used to load DataSets from a given Path (str) to a DataSet object
    - (serialized with (save-ds DataSet))
    - not yet implemented at time of writing doc string
   - i.e., RDD<String> to RDD<DataSet>"
  []
  (ds-fns {:path-to-ds {}}))

(defn new-path-to-multi-ds-fn
  "Simple function used to load MultiDataSets from a given Path (str) to a MultiDataSet object
    - (serialized with (save-ds MultiDataSet))
    - not yet implemented at time of writing doc string
   - i.e., RDD<String> to RDD<MultiDataSet>"
  []
  (ds-fns {:path-to-multi-ds {}}))

(defn new-split-ds-fn
  "Take an existing DataSet object, and split it into multiple DataSet objects
  with one example in each"
  []
  (ds-fns {:split-ds {}}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; shared fn from spark
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn call!
  "Calls one of the spark ds-fns on the dataset contained within the iter

  :ds-fn (obj), An instance of one of the ds-fn's defined in this ns
   - can also be a config map which gets passed to the ds-fns multi-method
     to create the ds-fn obj

  :iter (iterator), a dataset iterator
   - see: dl4clj.datasets.iterator.iterators and/or
          dl4clj.datasets.datavec

  returns a map of the ds-fn and iter"
  [& {:keys [ds-fn iter]}]
  ;; either way returns nil
  (if (map? ds-fn)
    (let [f (ds-fns ds-fn)]
      (do (.call f iter)
          {:fn-called f
           :called-on iter}))
    (do (.call ds-fn iter)
        {:fn-called ds-fn
         :called-on iter})))
