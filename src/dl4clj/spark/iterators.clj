(ns ^{:doc "spark dataset iterators, see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/iterator/package-summary.html"}
    dl4clj.spark.iterators
  (:import [org.deeplearning4j.spark.iterator
            PathSparkDataSetIterator
            PathSparkMultiDataSetIterator
            PortableDataStreamDataSetIterator
            PortableDataStreamMultiDataSetIterator])
  (:require [dl4clj.utils :refer [generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multimethod constructor calling
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti spark-stream-iterators generic-dispatching-fn)

(defmethod spark-stream-iterators :path-to-ds [opts]
  (let [conf (:path-to-ds opts)
        {str-paths :string-paths
         iter :iter} conf]
    (if (contains? conf :string-paths)
      (PathSparkDataSetIterator. str-paths)
      (PathSparkDataSetIterator. iter))))

(defmethod spark-stream-iterators :path-to-multi-ds [opts]
  (let [conf (:path-to-multi-ds opts)
        {str-paths :string-paths
         iter :iter} conf]
    (if (contains? conf :string-paths)
      (PathSparkMultiDataSetIterator. str-paths)
      (PathSparkMultiDataSetIterator. iter))))

(defmethod spark-stream-iterators :portable-ds-stream [opts]
  (let [conf (:portable-ds-stream opts)
        {streams :streams
         iter :iter} conf]
    (if (contains? conf :streams)
      (PortableDataStreamDataSetIterator. streams)
      (PortableDataStreamDataSetIterator. iter))))

(defmethod spark-stream-iterators :portable-multi-ds-stream [opts]
  (let [conf (:portable-multi-ds-stream opts)
        {streams :streams
         iter :iter} conf]
    (if (contains? conf :streams)
      (PortableDataStreamMultiDataSetIterator. streams)
      (PortableDataStreamMultiDataSetIterator. iter))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-path-spark-ds-iterator
  "A DataSetIterator that loads serialized DataSet objects
  from a String that represents the path
   -note: DataSet objects saved from a DataSet to an output stream

  :string-paths (coll), a collection of string paths representing the location
   of the data-set streams

  :iter (java.util.Iterator), an iterator for a collection of string paths
   representing the location of the data-set streams

  you should supply either :string-paths or :iter, if you supply both, will default
   to using the collection of string paths

  see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/iterator/PathSparkDataSetIterator.html"
  [& {:keys [string-paths iter]
      :as opts}]
  (spark-stream-iterators {:path-to-ds opts}))

(defn new-path-spark-multi-ds-iterator
  "A DataSetIterator that loads serialized MultiDataSet objects
  from a String that represents the path
   -note: DataSet objects saved from a MultiDataSet to an output stream

  :string-paths (coll), a collection of string paths representing the location
   of the data-set streams

  :iter (java.util.Iterator), an iterator for a collection of string paths
   representing the location of the data-set streams

  you should supply either :string-paths or :iter, if you supply both, will default
   to using the collection of string paths

  see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/iterator/PathSparkMultiDataSetIterator.html"
  [& {:keys [string-paths iter]
      :as opts}]
  (spark-stream-iterators {:path-to-multi-ds opts}))

(defn new-spark-portable-datastream-ds-iterator
  "A DataSetIterator that loads serialized DataSet objects
  from a PortableDataStream, usually obtained from SparkContext.binaryFiles()
   -note: DataSet objects saved from a DataSet to an output stream

  :streams (coll), a collection of portable datastreams

  :iter (java.util.Iterator), an iterator for a collection of portable datastreams

  you should only supply :streams or :iter, if you supply both, will default to
   using the collection of portable datastreams

  see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/iterator/PortableDataStreamDataSetIterator.html"
  [& {:keys [streams iter]
      :as opts}]
  (spark-stream-iterators {:portable-ds-stream opts}))

(defn new-spark-portable-datastream-multi-ds-iterator
  "A DataSetIterator that loads serialized MultiDataSet objects
  from a PortableDataStream, usually obtained from SparkContext.binaryFiles()
   -note: DataSet objects saved from a MultiDataSet to an output stream

  :streams (coll), a collection of portable datastreams

  :iter (java.util.Iterator), an iterator for a collection of portable datastreams

  you should only supply :streams or :iter, if you supply both, will default to
   using the collection of portable datastreams

  see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/iterator/PortableDataStreamMultiDataSetIterator.html"
  [& {:keys [streams iter]
      :as opts}]
  (spark-stream-iterators {:portable-multi-ds-stream opts}))
