(ns ^{:doc "see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/impl/common/repartition/BalancedPartitioner.html"}
    dl4clj.spark.common.balanced-partitioner
  (:import [org.deeplearning4j.spark.impl.common.repartition BalancedPartitioner]))

(defn new-balanced-partitioner
  ;; dont think this is a user facing fn
  ;; will be removed in core branch unless tests show otherwise
  "This is a custom partitioner which attempts to keep contiguous elements
  (i.e., those elements originally in the same partition) together much more frequently.
  Furthermore, it is less prone to producing larger or smaller than
  expected partitions, as it is entirely deterministic

  :n-partitions (int), desired number of partitions

  :elements-per-partition (int), the number of elements within each partition

  :remainder (int), expected number of stray elements"
  [& {:keys [n-partitions elements-per-partition remainder]}]
  (BalancedPartitioner. n-partitions elements-per-partition remainder))

(defn get-partition-by-key
  "returns the partition int identifier by the partitions key

  :partition-key (object), the key which identifies a partition
   - I think, not sure, will need to test"
  [& {:keys [balanced-partioner partition-key]}]
  (.getPartition balanced-partioner partition-key))

(defn get-number-of-partitions
  "return the number of partitions in a balanced partitioner"
  [balanced-partitioner]
  (.numPartitions balanced-partitioner))
