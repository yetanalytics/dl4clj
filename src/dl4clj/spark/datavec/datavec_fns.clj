(ns dl4clj.spark.datavec.datavec-fns
  (:import [org.deeplearning4j.spark.datavec
            DataVecSequencePairDataSetFunction
            DataVecSequenceDataSetFunction
            DataVecDataSetFunction
            RecordReaderFunction
            RDDMiniBatches
            RDDMiniBatches$MiniBatchFunction
            DataVecByteDataSetFunction])
  (:require [dl4clj.utils :refer [generic-dispatching-fn contains-many?]]
            [dl4clj.constants :as enum]))

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
    (cond (contains-many? conf :record-reader :label-idx :n-labels :writable-converter)
          (RecordReaderFunction. rr label-idx n-labels converter)
          (contains-many? conf :record-reader :label-idx :n-labels)
          (RecordReaderFunction. rr label-idx n-labels)
          :else
          (assert false "you must atleast supply a record reader, the idx of the label column
and the number of classes (labels)"))))

(defmethod datavec-fns :byte-ds [opts]
  (let [conf (:byte-ds opts)
        {label-idx :label-idx
         n-labels :n-labels
         batch-size :batch-size
         byte-file-len :byte-file-len
         regression? :regression?
         pp :pre-processor} conf]
    (cond (contains-many? conf :label-idx :n-labels :batch-size :btye-file-len :regression? :pre-processor)
          (DataVecByteDataSetFunction. label-idx n-labels batch-size byte-file-len regression? pp)
          (contains-many? conf :label-idx :n-labels :batch-size :btye-file-len :regression?)
          (DataVecByteDataSetFunction. label-idx n-labels batch-size byte-file-len regression?)
          (contains-many? conf :label-idx :n-labels :batch-size :btye-file-len)
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns, need to document keyword combos in doc string
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-record-reader-fn
  "Turn a string into a dataset based on a record reader
   - entry point for creating a dataset iterator in the context of spark

  :record-reader (record-reader), reads in the string based on its format
   - see: datavec.api.records.readers

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
   - see: nd4clj.linalg.dataset.api.pre-processors
   - clean the data before it is used

  the obj that is returned by this fn should be passed to call-datavec-fn! to be used"
  [& {:keys [label-idx n-labels batch-size byte-file-len regression? pre-processor]
      :as opts}]
  (datavec-fns {:byte-ds opts}))

(defn new-datavec-ds-fn
  "Transformation of writable objects (from a spark record reader) to dataset
  objects for spark training.

  Analogous to new-record-reader-dataset-iterator but in the context of spark.
   - dl4clj.datasets.datavec

  :label-idx (int), Idx of the label column

  :n-labels (int), number of classes for classification
   - not used if regression? set to true

  :regression? (boolean), is this a regression data set?
   - false = classification task

  :pre-processor (pre-processor), a dataset pre-processor
   - see: nd4clj.linalg.dataset.api.pre-processors
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
   - dl4clj.datasets.datavec
   - Supports loading data from a single source only

  :label-idx (int), Idx of the label column

  :n-labels (int), number of classes for classification
   - not used if regression? set to true

  :regression? (boolean), is this a regression data set?
   - false = classification task

  :pre-processor (pre-processor), a dataset pre-processor
   - see: nd4clj.linalg.dataset.api.pre-processors
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
  - dl4clj.datasets.datavec
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
   - see: nd4clj.linalg.dataset.api.pre-processors
   - clean the data before it is used

  :writable-converter (writable converter), converts a writable to another data type
   - need to have a central source of their creation
   - classes which implement the writable interface
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/Writable.html"
  [& {:keys [n-labels regression? spark-alignment-mode pre-processor writable-converter]
      :as opts}]
  (datavec-fns {:spark-seq-pair-ds-iter opts}))


(defn call-datavec-fns!
  "Calls one of the spark ds-fns on the dataset contained within the iter

  :datavec-fn (obj or map), either the return value of one of the new-datavec...-fn
   or a config map used to call the multi method.  see the fn doc strings for opts values
   and the fn body for fn type (first keyword, ie :spark-seq-ds-iter)
    - {:spark-seq-ds-iter {...opts}}

  :fn-dependent-data (data), data used in the call to the datavec-fn instance
   - when datavec-fn = record-reader-fn, data should be a string

   - when datavec-fn = new-datavec-byte-ds-fn, data should be:
     scala.Tuple2<hadoop.io.Text, hadoop.io.BytesWritable>

   - when datavec-fn = new-datavec-ds-fn, data should be:
     list of writables

   - when datavec-fn = new-datavec-seq-ds-fn, data should be:
     (list (list writable) (list writable) ...)

   -when datavec-fn = new-datavec-seq-pair-ds-fn, data should be:
     scala.Tuple2 <(list (list writable) (list writable)...), (list (list writable) (list writable) ...)>

  STILL NEED TO FIGURE OUT THE RDDMINIBATCHES CALL TYPE (3 subtypes)"
  [& {:keys [datavec-fn fn-dependent-data]}]
  ;; either way returns nil
  (if (map? datavec-fn)
    (let [f (datavec-fns datavec-fn)]
      (.call f fn-dependent-data))
    (.call datavec-fn fn-dependent-data)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; the subclass has the mini-batches-fn which has the call method
;; need to test to determine if i need to use that or if it gets called
;; when calling the superclass constructor
;; for now just going to implement the superclass constructor
;; https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/spark/dl4j-spark/src/main/java/org/deeplearning4j/spark/datavec/RDDMiniBatches.java
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