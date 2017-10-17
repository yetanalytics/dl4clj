(ns dl4clj.spark.utils
  (:import [org.deeplearning4j.spark.impl.graph.dataset DataSetToMultiDataSetFn]
           [org.deeplearning4j.spark.util SparkUtils])
  (:require [dl4clj.constants :refer [value-of]]))

;; update to work with code
;; WIP

(defn convert-ds-to-multi-ds
  "Convert a JavaRDD<DataSet> to a JavaRDD<MultiDataSet>"
  [data-set]
  (.call (DataSetToMultiDataSetFn.) data-set))

(defn balanced-random-split
  "Random split the specified RDD into a number of RDDs,
  where each has :n-objs-per-split in them.

  :total-obj-count (int), the total number of objects in the rdd or pair-rdd

  :n-objs-per-split (int), the desired number of dataset objects in each rdd or pair-rdd

  :pair-rdd (org.apache.spark.api.java.JavaPairRDD), a java rdd containing pairs of data
   to be split

  :seed (long), value for controling the RNG seed

  :rdd (org.apache.spark.api.java.JavaRDD), a java rdd containing the data you want split

  you must supply either :rdd or :pair-rdd based on the type of rdd containing the data
   - if for some reason you supply both, the rdd will be used"
  [& {:keys [total-obj-count n-objs-per-split
             pair-rdd seed rdd]
      :as opts}]
  (if (contains? opts :rdd)
    (if (contains? opts :seed)
      (.balancedRandomSplit total-obj-count n-objs-per-split rdd seed)
      (.balancedRandomSplit total-obj-count n-objs-per-split rdd))
    (if (contains? opts :seed)
      (.balancedRandomSplit total-obj-count n-objs-per-split pair-rdd seed)
      (.balancedRandomSplit total-obj-count n-objs-per-split pair-rdd))))

#_(defn check-kryo-config
  "Check the spark configuration for incorrect Kryo configuration,
  logging a warning message if necessary

  :java-spark-context (org.apache.spark.api.java.JavaSparkContext), the spark context

  :logger (org.slf4j.Logger), the logger to use for the warning message"
  [& {:keys [java-spark-context logger]}])

(defn list-paths-spark
  "List of the files in the given directory (path), as a JavaRDD<String>

  :java-spark-context (org.apache.spark.api.java.JavaSparkContext), the spark context

  :path (str), the path to the directory in question"
  [& {:keys [java-spark-context path]}]
  (.listPaths java-spark-context path))

(defn read-object-from-file
  "Read an object from HDFS (or local) using default Java object serialization

  :path (str), the path to the file in question

  :objt-type (java class), the type of the object being read in

  :spark-context (spark), can be a java-spark-context or a spark context
   - org.apache.spark.api.java.JavaSparkContext or org.apache.spark.SparkContext"
  [& {:keys [path obj-type spark-context]}]
  (.readObjectFromFile path obj-type spark-context))

(defn read-string-from-file
  "Read a UTF-8 format String from HDFS (or local)

  :path (str), the path to the file in question

  :spark-context (spark), can be a java-spark-context or a spark context
   - org.apache.spark.api.java.JavaSparkContext or org.apache.spark.SparkContext"
  [& {:keys [path spark-context]}]
  (.readStringFromFile path spark-context))

(defn repartition!
  "Repartition the specified RDD (or not) using the given
  :repartition and :repartition-strategy settings

  :rdd (JavaRDD), the data to be repartitioned

  :repartition (keyword), one of: :always, :never, or :num-partitions-workers-differs

  :repartition-strategy (keyword), one of: :balanced or :spark-default

  :objs-per-partition (int), desired number of objects per partition

  :n-partitions (int), desired number of partitions"
  [& {:keys [rdd repartition repartition-strategy objs-per-partition n-partitions]}]
  (.repartition rdd (value-of {:repartition repartition})
                (value-of {:repartition-strategy repartition-strategy})
                objs-per-partition n-partitions))

(defn repartition-balance-if-required!
  "Repartition a RDD (given the Repartition setting) such that we have
  approximately :n-partitions partitions, each of which has :objs-per-partition objects.

  :rdd (JavaRDD), the data to be repartitioned

  :repartition (keyword), one of: :always, :never, or :num-partitions-workers-differs

  :objs-per-partition (int), desired number of objects per partition

  :n-partitions (int), desired number of partitions"
  [& {:keys [rdd repartition objs-per-partition n-partitions]}]
  (.repartitionBalanceIfRequired rdd (value-of {:repartition repartition})
                                 objs-per-partition n-partitions))

(defn shuffle-examples!
  "Randomly shuffle the examples in each DataSet object,
  and recombine them into new DataSet objects with the specified :new-batch-size

  :rdd (JavaRDD<org.nd4j.linalg.dataset.DataSet>), the rdd containing the ds to be shuffled

  :new-batch-size (int), the desired new batch size for the datasets

  :n-partitions (int), the number of partitions within the rdd"
  [& {:keys [rdd new-batch-size n-partitions]}]
  (.shuffleExamples rdd new-batch-size n-partitions))

(defn write-obj-to-file!
  "Write an object to HDFS (or local) using default Java object serialization

  :path (str), the path to the file in question

  :the-obj (object), the object to write to the specified file

  :spark-context (spark), can be a java-spark-context or a spark context
   - org.apache.spark.api.java.JavaSparkContext or org.apache.spark.SparkContext

  returns the spark-context"
  [& {:keys [path the-obj spark-context]}]
  (SparkUtils/writeObjectToFile path the-obj spark-context)
  spark-context)

(defn write-string-to-file!
  "Write a String to a file (on HDFS or local) in UTF-8 format

  :path (str), the path to the file in question

  :the-str (str), the string to write to the specified file

  :spark-context (spark), can be a java-spark-context or a spark context
   - org.apache.spark.api.java.JavaSparkContext or org.apache.spark.SparkContext

  returns the spark-context"
  [& {:keys [path the-str spark-context]}]
  (.writeStringToFile path the-str spark-context)
  spark-context)
