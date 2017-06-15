(ns dl4clj.spark.common.load-serialized-ds
  (:import [org.deeplearning4j.spark.impl.common LoadSerializedDataSetFunction]
           [org.apache.spark.input PortableDataStream]
           [org.apache.hadoop.mapreduce.lib.input CombineFileSplit]
           [org.apache.hadoop.fs Path]))

(defn create-portable-data-stream
  "Portable datastreams allow DataStreams to be serialized and
   moved around by not creating them until they need to be read"
  ;; got super deep into the rabbit hole here
  ;; need to have a better understanding of spark/hadoop before can be used
  ;; https://github.com/damballa/parkour/tree/master
  ;; https://github.com/alexott/clojure-hadoop/tree/master
  ;; https://spark.apache.org/docs/2.1.0/api/java/org/apache/spark/input/PortableDataStream.html
  [& {:keys [combine-file-split hadoop-task-attempt-context idx]}]
  (PortableDataStream. combine-file-split hadoop-task-attempt-context idx))

;; for creating combine-file-splits
;; https://hadoop.apache.org/docs/r2.4.1/api/org/apache/hadoop/mapreduce/lib/input/CombineFileSplit.html
;; https://hadoop.apache.org/docs/r2.4.1/api/org/apache/hadoop/fs/Path.html

(defn load-serialized-dataset
  "This is a function that is used to load a DataSet object

  portable-data-stream (org.apache.spark.input.PortableDataStream)
   - the spark input stream of data to be converted into a dataset"
  [portable-data-stream]
  (.call (LoadSerializedDataSetFunction.) portable-data-stream))
