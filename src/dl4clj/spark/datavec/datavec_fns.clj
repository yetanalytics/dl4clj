(ns dl4clj.spark.datavec.datavec-fns
  (:import [org.deeplearning4j.spark.datavec
            DataVecSequencePairDataSetFunction
            DataVecSequenceDataSetFunction
            DataVecDataSetFunction
            RecordReaderFunction
            RDDMiniBatches
            RDDMiniBatches$MiniBatchFunction
            DataVecByteDataSetFunction]
           [org.deeplearning4j.spark.datavec.export StringToDataSetExportFunction])
  (:require [dl4clj.utils :refer [generic-dispatching-fn contains-many?]]
            [dl4clj.constants :as enum]
            [dl4clj.datasets.api.record-readers :refer [reset-rr!]]
            [dl4clj.helpers :refer [reset-iterator!]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method constructor calling
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti datavec-fns generic-dispatching-fn)

(defmethod datavec-fns :record-reader-fn [opts]
  (let [conf (:record-reader-fn opts)
        {rr :record-reader
         label-idx :label-idx
         n-labels :n-labels
         converter :writable-converter} conf]
    (let [r (reset-rr! rr)]
     (cond (contains-many? conf :record-reader :label-idx :n-labels :writable-converter)
          (RecordReaderFunction. r label-idx n-labels converter)
          (contains-many? conf :record-reader :label-idx :n-labels)
          (RecordReaderFunction. r label-idx n-labels)
          :else
          (assert false "you must atleast supply a record reader, the idx of the label column
and the number of classes (labels)")))))

(defmethod datavec-fns :byte-ds [opts]
  (let [conf (:byte-ds opts)
        {label-idx :label-idx
         n-labels :n-labels
         batch-size :batch-size
         byte-file-len :byte-file-len
         regression? :regression?
         pp :pre-processor} conf]
    (cond (contains-many? conf :label-idx :n-labels :batch-size :byte-file-len :regression? :pre-processor)
          (DataVecByteDataSetFunction. label-idx n-labels batch-size byte-file-len regression? pp)
          (contains-many? conf :label-idx :n-labels :batch-size :byte-file-len :regression?)
          (DataVecByteDataSetFunction. label-idx n-labels batch-size byte-file-len regression?)
          (contains-many? conf :label-idx :n-labels :batch-size :byte-file-len)
          (DataVecByteDataSetFunction. label-idx n-labels batch-size byte-file-len)
          :else
          ;; this will turn into a spec
          (assert false "you must atleast supply the idx of the label column,
 the number of classes (labels), the size of the ds and the number of bytes per individual file"))))

(defmethod datavec-fns :spark-ds-iter [opts]
  (let [conf (:spark-ds-iter opts)
        {label-idx :label-idx
         n-labels :n-labels
         regression? :regression?
         pp :pre-processor
         converter :writable-converter
         l-idx-from :label-idx-from
         l-idx-to :label-idx-to} conf]
    (cond (contains-many? conf :label-idx-from :label-idx-to :n-labels
                          :regression? :pre-processor :writable-converter)
          (DataVecDataSetFunction. l-idx-from l-idx-to n-labels regression? pp converter)
          (contains-many? conf :label-idx :n-labels :regression? :pre-processor :writable-converter)
          (DataVecDataSetFunction. label-idx n-labels regression? pp converter)
          (contains-many? conf :label-idx :n-labels :regression?)
          (DataVecDataSetFunction. label-idx n-labels regression?)
          :else
          (assert false "you must atleast supply the idx of the label column,
 the number of clsses (labels), and if the ds is for regression or classification"))))

(defmethod datavec-fns :spark-seq-ds-iter [opts]
  (let [conf (:spark-seq-ds-iter opts)
        {label-idx :label-idx
         n-labels :n-labels
         regression? :regression?
         pp :pre-processor
         converter :writable-converter} conf]
    (cond (contains-many? conf :label-idx :n-labels :regression? :pre-processor :writable-converter)
          (DataVecSequenceDataSetFunction. label-idx n-labels regression? pp converter)
          (contains-many? conf :label-idx :n-labels :regression?)
          (DataVecSequenceDataSetFunction. label-idx n-labels regression?)
          :else
          (assert false "you must atleast supply the idx of the label column,
the number of classes (labels), and if the ds is for regression or classification"))))

(defmethod datavec-fns :spark-seq-pair-ds-iter [opts]
  (let [conf (:spark-seq-pair-ds-iter opts)
        {n-labels :n-labels
         regression? :regression?
         s-a-mode :spark-alignment-mode
         pp :pre-processor
         converter :writable-converter} conf]
    (cond (contains-many? conf :n-labels :regression? :spark-alignment-mode
                          :pre-processor :writable-converter)
          (DataVecSequencePairDataSetFunction. n-labels regression?
                                               (enum/value-of {:spark-alignment-mode s-a-mode})
                                               pp converter)
          (contains-many? conf :n-labels :regression? :spark-alignment-mode)
          (DataVecSequencePairDataSetFunction. n-labels regression?
                                               (enum/value-of {:spark-alignment-mode s-a-mode}))
          (contains-many? conf :n-labels :regression?)
          (DataVecSequencePairDataSetFunction. n-labels regression?)
          :else
          (DataVecSequencePairDataSetFunction.))))

(defmethod datavec-fns :string-to-ds-export [opts]
  (let [conf (:string-to-ds-export opts)
        {dir :output-directory
         rr :record-reader
         batch-size :batch-size
         r? :regression?
         l-idx :label-idx
         n-labels :n-labels} conf]
    (StringToDataSetExportFunction. (java.net.URI/create dir) (reset-rr! rr) batch-size r? l-idx n-labels)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns, need to document keyword combos in doc string
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-record-reader-fn
  "Turn a string into a dataset based on a record reader
   - entry point for creating a dataset iterator in the context of spark

  :record-reader (record-reader), reads in the string based on its format
   - see: dl4clj.datasets.record-readers

  :label-idx (int), Idx of the label column

  :n-labels (int), number of classes for setting up the dataset

  :writable-converter (writable converter), converts a writable to another data type
   - need to have a central source of their creation
   - classes which implement the writable interface
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/Writable.html"
  [& {:keys [record-reader label-idx n-labels writable-converter]
      :as opts}]
  (datavec-fns {:record-reader-fn opts}))

(defn new-datavec-byte-ds-fn
  "spark dataset fn for setting up a byte ds

  :label-idx (int), Idx of the label column

  :n-labels (int), number of classes for classification
   - not used if regression? set to true

  :batch-size (int), size of a subset of examples from the dataset.
   - pass in the total number of examples if you want to include the whole ds

  :byte-file-len (int), number of pytes per individual file

  :regression? (boolean), is this a regression data set?
   - false = classification task

  :pre-processor (pre-processor), a dataset pre-processor
   - see: dl4clj.datasets.pre-processors
   - clean the data before it is used

  the obj that is returned by this fn should be passed to call-datavec-fn! to be used"
  [& {:keys [label-idx n-labels batch-size byte-file-len regression? pre-processor]
      :as opts}]
  (datavec-fns {:byte-ds opts}))

(defn new-datavec-ds-fn
  "Transformation of writable objects (from a spark record reader) to dataset
  objects for spark training.

  Analogous to new-record-reader-dataset-iterator but in the context of spark.
   - dl4clj.datasets.iterators

  :label-idx (int), Idx of the label column

  :n-labels (int), number of classes for classification
   - not used if regression? set to true

  :regression? (boolean), is this a regression data set?
   - false = classification task

  :pre-processor (pre-processor), a dataset pre-processor
   - see: dl4clj.datasets.pre-processors
   - clean the data before it is used

  :writable-converter (writable converter), converts a writable to another data type
   - need to have a central source of their creation
   - classes which implement the writable interface
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/Writable.html

  :label-idx-from (int), idx of the first target

  :label-idx-to (int), idx of the last target
   - inclusive
   - for classification or single-output regression, same as label-idx-from"
  [& {:keys [label-idx n-labels regression? pre-processor
             writable-converter label-idx-from label-idx-to]
      :as opts}]
  (datavec-fns {:spark-ds-iter opts}))

(defn new-datavec-seq-ds-fn
  "Transforming a collection of writable objects (from a spark seq record reader)
   to a dataset obj for spark training.

  Analogous to new-seq-record-reader-dataset-iterator but in the context of spark.
   - dl4clj.datasets.iterators
   - Supports loading data from a single source only

  :label-idx (int), Idx of the label column

  :n-labels (int), number of classes for classification
   - not used if regression? set to true

  :regression? (boolean), is this a regression data set?
   - false = classification task

  :pre-processor (pre-processor), a dataset pre-processor
   - see: dl4clj.datasets.pre-processors
   - clean the data before it is used

  :writable-converter (writable converter), converts a writable to another data type
   - need to have a central source of their creation
   - classes which implement the writable interface
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/Writable.html"
  [& {:keys [label-idx n-labels regression? pre-processor writable-converter]
      :as opts}]
  (datavec-fns {:spark-seq-ds-iter opts}))

(defn new-datavec-seq-pair-ds-fn
  "Transform a tuple of collections of writables (out of 2 datavec-spark seq rr fns)
  to dataset objects for spark training

  Analogous to new-seq-record-reader-dataset-iterator but in the context of spark.
  - dl4clj.datasets.iterators
  - Supports loading data from a two sources only
    - supports many-to-one and one-to-many situations

  :n-labels (int), number of classes for classification
   - not used if regression? is set to true

  :regression? (boolean), is this a regression data set?
   - false = classification task

  :spark-alignment-mode (keyword), Alignment mode for dealing with input/labels
   of differing lengths (one-to-many and many-to-one type situations)
   - one of :align-end, :align-start, :equal-length

  :pre-processor (pre-processor), a dataset pre-processor
   - see: dl4clj.datasets.pre-processors
   - clean the data before it is used

  :writable-converter (writable converter), converts a writable to another data type
   - need to have a central source of their creation
   - classes which implement the writable interface
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/Writable.html"
  [& {:keys [n-labels regression? spark-alignment-mode pre-processor writable-converter]
      :as opts}]
  (datavec-fns {:spark-seq-pair-ds-iter opts}))

(defn new-string-to-ds-export-fn
  "A function used in forEachPartition to convert Strings to Dataset Objects
   using a record reader.

  :output-directory (str), the directory to export to

  :record-reader (record-reader), the object containing the string data
   - see: dl4clj.datasets.record-readers

  :batch-size (int), the size of the minibatch

  :regression? (boolean), are we dealing with a classification or regression dataset

  :label-idx (int), the column index of the labels within the record-reader

  :n-labels (int), the total number of possible labels or classes"
  [& {:keys [output-directory record-reader batch-size
             regression? label-idx n-labels]
      :as opts}]
  (datavec-fns {:string-to-ds-export opts}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; calling fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn call-record-reader-fn!
  "uses the object created by new-record-reader-fn to perform its function
   - turn a string into a dataset based on a record reader

  :the-fn (obj or map), will accept the object created using new-record-reader-fn,
   but will also accept a config map used to call the datavec-fns multimethod
   directly.
    - config map = {:record-reader-fn {:record-reader (record-reader) :label-idx (int)
                                       :n-labels (int), :writable-converter (writable-converter)}}
    - only :record-reader, :label-idx and :n-labels are required

  :string-ds (str), the dataset as a string to be used in a string-split
   - for info on string splits, see: datavec.api.split"
  [& {:keys [the-fn string-ds]}]
  (if (map? the-fn)
    (.call (datavec-fns the-fn) string-ds)
    (.call the-fn string-ds)))

(defn call-datavec-byte-ds-fn!
  "uses the object created by new-datavec-byte-ds-fn to perform its function
   - turn a string into a dataset based on a record reader

  :the-fn (obj or map), will accept the object created using new-datavec-byte-ds-fn,
   but will also accept a config map used to call the datavec-fns multimethod
   directly.
    - config map = {:byte-ds {:batch-size (int) :label-idx (int) :n-labels (int),
                              :byte-file-len (int), :regression? (boolean),
                              :pre-processor (pre-processor)}}
    - only :label-idx, :n-labels, :batch-size, :byte-file-len are required

  :input-tuple (scala.Tuple2) <hadoop.io.Text, hadoop.io.BytesWritable>
   - see: https://deeplearning4j.org/datavecdoc/org/datavec/spark/functions/data/FilesAsBytesFunction.html
          - needs to be implemented"
  [& {:keys [the-fn input-tuple]}]
  (if (map? the-fn)
    (.call (datavec-fns the-fn) input-tuple)
    (.call the-fn input-tuple)))

(defn call-datavec-ds-fn!
  "uses the object created by new-datavec-ds-fn to perform its function
   - maps a collection of writable objects (out of a datavec-spark record reader fn)
     to DataSet objects for Spark training

  :the-fn (obj or map), will accept the object created using new-datavec-ds-fn,
   but will also accept a config map used to call the datavec-fns multimethod
   directly.
    - config map = {:spark-ds-iter {:label-idx (int), :n-labels (int) :regression? (boolean)
                                    :pre-processor (pre-processor), :writable-converter (writable)
                                    :label-idx-from (int), :label-idx-to (int)}}
    - only :label-idx, :n-labels, :regression? are required

  :list-of-writables (list), a collection (list), of writable objects
   - classes which implement the writable interface
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/Writable.html
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/package-tree.html"
  [& {:keys [the-fn list-of-writables]}]
  (if (map? the-fn)
    (.call (datavec-fns the-fn) list-of-writables)
    (.call the-fn list-of-writables)))

(defn call-datavec-seq-ds-fn!
  "uses the object created by new-datavec-seq-ds-fn to perform its function
   - maps a collection of collections of writables (out of a datavec-spark sequence record reader fn)
     to DataSet objects for spark training

  :the-fn (obj or map), will accept the object created using new-datavec-seq-ds-fn,
   but will also accept a config map used to call the datavec-fns multimethod directly.
    - config map = {:spark-seq-ds-iter {:label-idx (int), :n-labels (int),
                                        :regression? (boolean), :pre-processor (pre-processor)
                                        :writable-converter (converter)}}
    - only :label-idx, :n-labels, :regression? are required

  :input (list), (list (list writable) (list writable) ...)
   - classes which implement the writable interface
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/Writable.html
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/package-tree.html"
  [& {:keys [the-fn input]}]
  (if (map? the-fn)
    (.call (datavec-fns the-fn) input)
    (.call the-fn input)))

(defn call-datavec-seq-pair-ds-fn!
  "uses the object created by new-datavec-seq-pair-ds-fn to perform its function
   - maps a tuple of collection of collections of writables (out of two datavec-spark sequence record reader fns)
     to DataSet objects for spark training

  :the-fn (obj or map), will accept the object created using new-datavec-seq-pair-ds-fn,
   but will also accept a config map used to call the datavec-fns multimethod directly.
    - config map = {:spark-seq-pair-ds-iter {:spark-alignment-mode (keyword), :n-labels (int),
                                             :regression? (boolean), :pre-processor (pre-processor)
                                             :writable-converter (converter)}}
    - only :label-idx, :n-labels, :regression? are required

  :input (scala.Tuple2),
  <(list (list writable) (list writable) ...), (list (list writable) (list writable) ...)>
   - see :https://deeplearning4j.org/datavecdoc/org/datavec/spark/functions/pairdata/PairSequenceRecordReaderBytesFunction.html
     - needs to be implemented, its how you create the scala.Tuple2 input
       - which needs https://deeplearning4j.org/datavecdoc/org/datavec/spark/functions/pairdata/BytesPairWritable.html
   - classes which implement the writable interface
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/Writable.html
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/package-tree.html"
  [& {:keys [the-fn input-tuple]}]
  (if (map? the-fn)
    (.call (datavec-fns the-fn) input-tuple)
    (.call the-fn input-tuple)))

(defn call-string-to-ds-export-fn!
  "uses the object created by new-string-to-ds-fn to perform its function
   - converts strings to datasset objects using a record reader

  :the-fn (obj or map), will  accept the object created using new-string-to-ds-export-fn
   but will also accept a config map used to call the datavec-fns multimethod directly.
   - config map = {:string-to-ds-export {:output-directory (str), :record-reader (record reader),
                                         :batch-size (int), :regression? (boolean),
                                         :label-idx (int), :n-labels (int)}}
   - all args are required to create the string-to-ds-export fn

  :iter (iterator), an iterator which contains the string data to convert and export
   - see: dl4clj.datasets.iterators

  returns the iterator and the fn-object"
  [& {:keys [the-fn iter]}]
  (let [ds-iter (reset-iterator! iter)]
   (if (map? the-fn)
    (let [f (datavec-fns the-fn)]
      (do (.call f ds-iter)
        {:fn f :iter ds-iter}))
    (do (.call the-fn ds-iter)
        {:fn the-fn :iter ds-iter}))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; the subclass has the mini-batches-fn which has the call method
;; need to test to determine if i need to use that or if it gets called
;; when calling the superclass constructor
;; for now just going to implement the superclass constructor
;; https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/spark/dl4j-spark/src/main/java/org/deeplearning4j/spark/datavec/RDDMiniBatches.java
;; come back to test once there is a good way of making java rdds out of datasets
(defmethod datavec-fns :rdd-mini-batches [opts]
  (let [conf (:rdd-mini-batches opts)
        {mini-batch :mini-batch
         to-split :to-split} conf]
    (RDDMiniBatches. mini-batch to-split)))

(defn new-rdd-mini-batches
  "RDD mini batch partitioning

  :mini-batch (int), the number of mini batches to make

  :to-split (rdd), a dataset contained within a java rdd"
  [& {:keys [mini-batch to-split]
      :as opts}]
  (datavec-fns {:rdd-mini-batches opts}))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
