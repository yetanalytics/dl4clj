(ns ^{:doc "name space for creating JavaRDDs, Not sure if this is the best way of doing it, need to do more research"}
    dl4clj.spark.data.java-rdd
  (:import [org.apache.spark.api.java JavaRDD JavaSparkContext]
           [org.apache.spark SparkConf])
  (:require [dl4clj.helpers :refer [data-from-iter reset-iterator!]]
            [dl4clj.utils :refer [obj-or-code?]]))

(defn new-java-spark-context
  "creates a java spark context from a spark conf

  if no spark conf is supplied, one will be create
   - master is will be set to local
   - app name will be set to :app-name or 'default app name'"
  [& {:keys [spark-conf app-name as-code?]
      :or {as-code? false}
      :as opts}]
  (let [code (if (contains? opts :spark-conf)
               `(JavaSparkContext. ~spark-conf)
               `(-> (SparkConf.)
                    (.setMaster "local[*]")
                    (.setAppName ~(if (string? app-name)
                                    app-name
                                    "default app name"))
                    (JavaSparkContext.)))]
    (obj-or-code? as-code? code)))

(defn text-file
  "Reads a text file from HDFS, a local file system (available on all nodes),
  or any Hadoop-supported file system URI, and returns it as an `JavaRDD` of Strings.

  :spark-context (spark context), your connection to spark

  :file-name (str) path to the file you want in the JavaRDD

  :min-partitions (int), the desired mininum number of partitions
   - optional"
  [& {:keys [spark-context file-name min-partitions]}]
  (if (int? min-partitions)
    (.textFile spark-context file-name min-partitions)
    (.textFile spark-context file-name)))

(defn whole-text-files
  "Read a directory of text files from HDFS, a local file system (available on all nodes),
  or any Hadoop-supported file system URI, and returns it as a `JavaPairRDD` of (K, V) pairs,
  where, K is the path of each file, V is the content of each file.

  :spark-context (spark context), your connection to spark

  :path (str) path to the directory

  :min-partitions (int), the desired mininum number of partitions
   - optional"
  [& {:keys [spark-context path min-partitions]}]
  (if (int? min-partitions)
    (.wholeTextFiles spark-context path min-partitions)
    (.wholeTextFiles spark-context path)))

(defn parallelize
  "Distributes a local collection to form/return an RDD

  :spark-context (spark context), your connection to spark

  :data (coll), a collection you want converted into a RDD

  :num-slices (int), need to get more familiar with spark lingo to give an accurate desc
   - optional"
  [& {:keys [spark-context data num-slices]}]
  (if (int? num-slices)
    (.parallelize spark-context data num-slices)
    (.parallelize spark-context data)))

(defn parallelize-pairs
  "Distributes a local collection to form/return a Pair RDD

  :spark-context (spark context), your connection to spark

  :data (coll), a collection you want converted into a RDD

  :num-slices (int), need to get more familiar with spark lingo to give an accurate desc
   - optional"
  [& {:keys [spark-context data num-slices]}]
  (if (int? num-slices)
    (.parallelizePairs spark-context data num-slices)
    (.parallelizePairs spark-context data)))

(defn java-rdd-from-iter
  "given a spark context and an iterator, creates a javaRDD from the
  data in the iterator"
  [& {:keys [spark-context iter num-slices as-code?]}]
  (let [data (data-from-iter (reset-iterator! iter))]
    (parallelize :spark-context spark-context
                 :data data
                 :num-slices num-slices)))
