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
            BatchDataSetsFunction]
           [org.deeplearning4j.spark.data.shuffle SplitDataSetExamplesPairFlatMapFunction])
  (:require [dl4clj.utils :refer [generic-dispatching-fn obj-or-code?]]
            [clojure.core.match :refer [match]]
            [dl4clj.helpers :refer [reset-iterator!]]))

;; need to update to default to code

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method for constructor calling
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti ds-fns generic-dispatching-fn)

(defmethod ds-fns :batch-and-export-ds [opts]
  (let [conf (:batch-and-export-ds opts)
        {batch-size :batch-size
         path :export-path} conf]
    `(BatchAndExportDataSetsFunction. ~batch-size ~path)))

(defmethod ds-fns :batch-and-export-multi-ds [opts]
  (let [conf (:batch-and-export-multi-ds opts)
        {batch-size :batch-size
         path :export-path} conf]
    `(BatchAndExportMultiDataSetsFunction. ~batch-size ~path)))

(defmethod ds-fns :batch-ds [opts]
  (let [mbs (:batch-size (:batch-ds opts))]
    `(BatchDataSetsFunction. ~mbs)))

(defmethod ds-fns :export-ds [opts]
  (let [p (:export-path (:export-ds opts))]
    `(DataSetExportFunction. (java.net.URI/create ~p))))

(defmethod ds-fns :export-multi-ds [opts]
  (let [p (:export-path (:export-multi-ds opts))]
    `(MultiDataSetExportFunction. (java.net.URI/create ~p))))

(defmethod ds-fns :split-ds [opts]
  `(SplitDataSetsFunction.))

(defmethod ds-fns :split-ds-rand [opts]
  (let [max-k (:max-key-idx (:split-ds-rand opts))]
    `(SplitDataSetExamplesPairFlatMapFunction. ~max-k)))

(defmethod ds-fns :path-to-ds [opts]
  `(PathToDataSetFunction.))

(defmethod ds-fns :path-to-multi-ds [opts]
  `(PathToMultiDataSetFunction.))

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
  [& {:keys [batch-size export-path as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (ds-fns {:batch-and-export-ds opts})]
    (obj-or-code? as-code? code)))

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
  [& {:keys [batch-size export-path as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (ds-fns {:batch-and-export-multi-ds opts})]
    (obj-or-code? as-code? code)))

(defn new-batch-ds-fn
  "Function used to batch DataSet objects together.

  Typically used to combine singe-example DataSet objects out of something
  like DataVecDataSetFunction together into minibatches.

  Usage: (def single-ds-ex (map-partitions (new-batch-ds-fn n)))

  :batch-size (int), the size of the batches"
  [& {:keys [batch-size as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (ds-fns {:batch-ds opts})]
    (obj-or-code? as-code? code)))

(defn new-ds-export-fn
  "A function to save DataSet objects to disk/HDFS.
   - used with (for-each-partition [data JavaRDD<DataSet>] (new-ds-export-fn export-path))
     - need to implement for-each-partition at time of writing doc string

  Each DataSet object is given a random and (probably) unique name starting with
  dataset_ and ending with .bin.

  :export-path (str), the place to export to"
  [& {:keys [export-path as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (ds-fns {:export-ds opts})]
    (obj-or-code? as-code? code)))

(defn new-multi-ds-export-fn
  "A function to save MultiDataSet objects to disk/HDFS.
   - used with (for-each-partition [data JavaRDD<MultiDataSet>] (new-ds-export-fn export-path))
     - need to implement for-each-partition at time of writing doc string

  Each MultiDataSet object is given a random and (probably) unique name starting with
  dataset_ and ending with .bin.

  :export-path (str), the place to export to"
  [& {:keys [export-path as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (ds-fns {:export-multi-ds opts})]
    (obj-or-code? as-code? code)))

(defn new-path-to-ds-fn
  "Simple function used to load DataSets from a given Path (str) to a DataSet object
    - (serialized with (save-ds DataSet))
    - not yet implemented at time of writing doc string
   - i.e., RDD<String> to RDD<DataSet>"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (ds-fns {:path-to-ds {}})]
    (obj-or-code? as-code? code)))

(defn new-path-to-multi-ds-fn
  "Simple function used to load MultiDataSets from a given Path (str) to a MultiDataSet object
    - (serialized with (save-ds MultiDataSet))
    - not yet implemented at time of writing doc string
   - i.e., RDD<String> to RDD<MultiDataSet>"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (ds-fns {:path-to-multi-ds {}})]
    (obj-or-code? as-code? code)))

(defn new-split-ds-fn
  "Take an existing DataSet object, and split it into multiple DataSet objects
  with one example in each"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (ds-fns {:split-ds {}})]
    (obj-or-code? as-code? code)))

(defn new-split-ds-with-appended-key
  "splits each example in a DataSet object into its own DataSet.

  Also adds a random key (int) in the range 0 to (- max-key-idx 1).

  :max-key-idx (int), used for adding random keys to the new datasets"
  [& {:keys [max-key-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (ds-fns {:split-ds-rand opts})]
    (obj-or-code? as-code? code)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; calling fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn call-batch-and-export-ds-fn!
  "uses the object created by new-batch-and-export-ds-fn
   to perform its function
    - batch DataSet objects together, and save the result

  :the-fn (obj or config-map), will accept the object created using
   new-batch-and-export-ds-fn, but will also accept a config map used
   to call the ds-fns multimethod directly
    - config map = {:batch-and-export-ds {:batch-size (int) :export-path (str)}}
      - opts is a map with keys/values described in new-batch-and-export-ds-fn

  :partition-idx (int), tag for labeling the new datasets created via this fn

  :iter (dataset iterator), the iterator which goes through a dataset
   - see: dl4clj.datasets.iterators"
  [& {:keys [the-fn partition-idx iter as-code?]
      :or {as-code? true}
      :as opts}]
  (match [(dissoc opts :as-code?)]
         [{:the-fn (_ :guard map?)
           :partition-idx (:or (_ :guard number?)
                               (_ :guard seq?))
           :iter (_ :guard seq?)}]
         (obj-or-code? as-code? `(.call ~(ds-fns the-fn) (int ~partition-idx) ~iter))
         [{:the-fn (_ :guard map?)
           :partition-idx (:or (_ :guard number?)
                               (_ :guard seq?))
           :iter _}]
         (.call (eval (ds-fns the-fn)) (int partition-idx) (reset-iterator! iter))
         [{:the-fn (_ :guard seq?)
           :partition-idx (:or (_ :guard number?)
                               (_ :guard seq?))
           :iter (_ :guard seq?)}]
         (obj-or-code? as-code? `(.call ~the-fn (int ~partition-idx) ~iter))
         [{:the-fn _
           :partition-idx (:or (_ :guard number?)
                               (_ :guard seq?))
           :iter _}]
         (.call the-fn (int partition-idx) (reset-iterator! iter))))

(defn call-batch-and-export-multi-ds-fn!
  "uses the object created by new-batch-and-export-multi-ds-fn
   to perform its function
    - batch MultiDataSet objects together, and save the result

  :the-fn (obj or config-map), will accept the object created using
   new-batch-and-export-ds-fn, but will also accept a config map used
   to call the ds-fns multimethod directly
    - config map = {:batch-and-export-multi-ds {:batch-size (int) :export-path (str)}}
      - opts is a map with keys/values described in new-batch-and-export-multi-ds-fn

  :partition-idx (int), tag for labeling the new datasets created via this fn

  :multi-ds-iter (multi-dataset iterator), the iterator which goes through a dataset
   - see: dl4clj.datasets.iterators"
  [& {:keys [the-fn partition-idx multi-ds-iter]}]
  (let [mds-iter (reset-iterator! multi-ds-iter)]
   (if (map? the-fn)
    (.call (ds-fns the-fn) (int partition-idx) mds-iter)
    (.call the-fn (int partition-idx) mds-iter))))

(defn call-batch-ds-fn!
  "uses the object created by new-batch-ds-fn
   to perform its function
    - batch DataSet objects together, ie. combine single-example ds objs
      together into minibatches

  :the-fn (obj or map), will accept the object created by new-batch-ds-fn
   or a config map to make that object
   - config map = {:batch-ds {:batch-size (int)}}

  :iter, (iterator), an iterator which wraps a dataset you want batched
   - see: dl4clj.datasets.iterators"
  [& {:keys [the-fn iter]}]
  (let [ds-iter (reset-iterator! iter)]
   (if (map? the-fn)
    (.call (ds-fns the-fn) ds-iter)
    (.call the-fn ds-iter))))

(defn call-ds-export-fn!
  "uses the object created by new-ds-export-fn to perform its function
   - save the dataset to a specified directory

  :the-fn (obj or map), will accept the object created by new-ds-export-fn
   or a config map to make the object
    - config mpa = {:export-ds {:export-path path (str)}}

  :iter (iterator), an iterator which wraps a dataset you want exported
   - see: dl4clj.datasets.iterators

  returns a map of the export-fn and ds-iter"
  [& {:keys [the-fn iter]}]
  (let [ds-iter (reset-iterator! iter)]
   (if (map? the-fn)
    (do (.call (ds-fns the-fn) ds-iter)
        {:export-fn the-fn :iter ds-iter})
    (do (.call the-fn ds-iter)
        {:export-fn the-fn :iter ds-iter}))))

(defn call-multi-ds-export-fn!
  "uses the object created by new-multi-ds-export-fn to perform its function
   - save the multi dataset to a specified directory

  :the-fn (obj or map), will accept the object created by new-multi-ds-export-fn
   or a config map to make the object
    - config mpa = {:export-multi-ds {:export-path path (str)}}

  :iter (iterator), an iterator which wraps a dataset you want exported
   - see: dl4clj.datasets.iterators

  returns a map of the export-fn and ds-iter"
  [& {:keys [the-fn iter]}]
  (let [ds-iter (reset-iterator! iter)]
   (if (map? the-fn)
    (do (.call (ds-fns the-fn) ds-iter)
        {:export-fn the-fn :iter ds-iter})
    (do (.call the-fn ds-iter)
        {:export-fn the-fn :iter ds-iter}))))

(defn call-path-to-ds-fn!
  "uses the object created by new-path-to-ds-fn to perform its function
   - load serialized datasets from a given path
   - RDD<String> to RDD<DataSet>

  :the-fn (obj or map), will accept the object created by new-path-to-ds-fn
   or a config map to make the object
    - config map = {:path-to-ds {}}

  :path (str), the path to the seralized statsets
   - needs to point at the ds file itself not the directory its in"
  [& {:keys [the-fn path]}]
  (if (map? the-fn)
    (.call (ds-fns the-fn) path)
    (.call the-fn path)))

(defn call-path-to-multi-ds-fn!
  "uses the object created by new-path-to-multi-ds-fn to perform its function
   - load serialized multi-datasets from a given path
   - RDD<String> to RDD<MultiDataSet>

  :the-fn (obj or map), will accept the object created by new-path-to-multi-ds-fn
   or a config map to make the object
    - config map = {:path-to-multi-ds {}}

  :path (str), the path to the seralized statsets
   - needs to point at the ds file itself not the directory its in"
  [& {:keys [the-fn path]}]
  (if (map? the-fn)
    (.call (ds-fns the-fn) path)
    (.call the-fn path)))

(defn call-split-ds-fn!
  "uses the object created by new-split-ds-fn to perform its function
   - split an existing dataset into multiple dataset objecs with one example each

  :the-fn (obj or map), will accept the object created by new-split-ds-fn
   or a config map to make the object
    - config map = {:split-ds {}}

  :iter (iterator), an iterator which wraps a dataset you want split
   - see: dl4clj.datasets.iterators"
  [& {:keys [the-fn iter]}]
  (let [ds-iter (reset-iterator! iter)]
   (if (map? the-fn)
    (.call (ds-fns the-fn) ds-iter)
    (.call the-fn ds-iter))))

(defn call-split-ds-with-appended-key!
  "uses the object created by new-split-ds-with-appended-key to perform its function
   - split an existing dataset into multiple dataset objecs with one example each
     and tagging those objs with a random key from 0 (- max-key-idx 1)

  :the-fn (obj or map), will accept the object created by new-split-ds-fn
   or a config map to make the object
    - config map = {:split-ds-rand {:max-key-idx (int)}}

  :ds (data-set), the dataset you want to split"
  [& {:keys [the-fn ds]}]
  (if (map? the-fn)
    (.call (ds-fns the-fn) ds)
    (.call the-fn ds)))
