(ns dl4clj.datasets.datavec
  (:import [org.deeplearning4j.datasets DataSets]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator
            RecordReaderMultiDataSetIterator$Builder RecordReaderMultiDataSetIterator
            SequenceRecordReaderDataSetIterator])
  (:require [dl4clj.constants :refer [value-of]]
            [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]
            ;; write mmethod for making writeable converters and require it here
            ;; https://deeplearning4j.org/datavecdoc/org/datavec/api/io/package-summary.html
            ;; they are going to be in datavec.api.io
            ))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; build in datasets
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def iris-ds (DataSets/iris))

(def mnist-ds (DataSets/mnist))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; iterator multimethod
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti iterator
  "Multimethod that builds a dataset iterator based on the supplied type and opts"
  generic-dispatching-fn)

(defmethod iterator :rr-dataset-iter [opts]
  (let [config (:rr-dataset-iter opts)
        {rr :record-reader
         batch-size :batch-size
         label-idx :label-idx
         n-labels :n-possible-labels
         l-idx-from :label-idx-from
         l-idx-to :label-idx-to
         regression? :regression?
         max-n-batches :max-num-batches
         converter :writeable-converter} config]
    (assert (contains-many? config :record-reader :batch-size)
            "you must supply atleast a record reader config map and a batch size")
    (if (contains? config :writeable-converter)
      (cond (contains-many? config :batch-size :label-idx-from :label-idx-to
                            :n-possible-labels :max-num-batches :regression?)
            (RecordReaderDataSetIterator.
             rr
             converter ;; write mmethod for making these
             batch-size l-idx-from l-idx-to n-labels
             max-n-batches regression?)
            (contains-many? config :batch-size :label-idx :n-possible-labels
                            :max-num-batches :regression?)
            (RecordReaderDataSetIterator.
             rr
             converter ;; write mmethod for making these
             batch-size label-idx n-labels max-n-batches regression?)
            (contains-many? config :batch-size :label-idx :n-possible-labels :regression?)
            (RecordReaderDataSetIterator.
             rr
             converter ;; write mmethod for making these
             batch-size label-idx n-labels regression?)
            (contains-many? :batch-size :label-idx :n-possible-labels)
            (RecordReaderDataSetIterator.
             rr
             converter ;; write mmethod for making these
             batch-size label-idx n-labels)
            (contains? config :batch-size)
            (RecordReaderDataSetIterator.
             rr
             converter ;; write mmethod for making these
             batch-size)
            :else
            (assert false "you must provide a record reader, writeable converter and a batch size"))
      (cond (contains-many? config :batch-size :label-idx :n-possible-labels :max-num-batches)
            (RecordReaderDataSetIterator.
             rr batch-size label-idx n-labels max-n-batches)
            (contains-many? config :batch-size :label-idx-from :label-idx-to :regression?)
            (RecordReaderDataSetIterator.
             rr batch-size l-idx-from l-idx-to regression?)
            (contains-many? config :batch-size :label-idx :n-possible-labels)
            (RecordReaderDataSetIterator.
             rr batch-size label-idx n-labels)
            (contains? config :batch-size)
            (RecordReaderDataSetIterator.
            rr batch-size)
            :else
            (assert false "you must supply a record reader and a batch size")))))

(defmethod iterator :seq-rr-dataset-iter [opts]
  (let [config (:seq-rr-dataset-iter opts)
        {rr :record-reader
         m-batch-size :mini-batch-size
         n-labels :n-possible-labels
         label-idx :label-idx
         regression? :regression?
         labels-reader :labels-reader
         features-reader :features-reader
         alignment :alignment-mode} config]
    (assert (or (and (contains? config :labels-reader)
                     (contains? config :features-reader))
                (contains? config :record-reader))
            "you must supply a record reader or a pair of labels/features readers")
    (if (contains-many? config :labels-reader :features-reader)
      (cond (contains-many? config :mini-batch-size :n-possible-labels
                            :regression? :alignment-mode)
            (SequenceRecordReaderDataSetIterator.
             features-reader labels-reader
             m-batch-size n-labels regression?
             (value-of {:seq-alignment-mode alignment}))
            (contains-many? config :mini-batch-size :n-possible-labels :regression?)
            (SequenceRecordReaderDataSetIterator.
             features-reader labels-reader
             m-batch-size n-labels regression?)
            (contains-many? config :mini-batch-size :n-possible-labels)
            (SequenceRecordReaderDataSetIterator.
             features-reader labels-reader
             m-batch-size n-labels)
            :else
            (assert false "if youre supplying seperate labels and features readers,
you must supply atleast a batch size and the number of possible labels"))
      (cond (contains-many? :mini-batch-size :n-possible-labels :label-idx
                            :regression?)
            (SequenceRecordReaderDataSetIterator.
             rr m-batch-size n-labels label-idx regression?)
            (contains-many? :mini-batch-size :n-possible-labels :label-idx)
            (SequenceRecordReaderDataSetIterator.
             rr m-batch-size n-labels label-idx)
            :else
            (assert false "if you're supplying a single record reader for the features and the labels,
you need to suply atleast the mini batch size, number of possible labels and the column index of the labels")))))

(defmethod iterator :multi-dataset-iter [opts]
  (assert (integer? (:batch-size (:multi-dataset-iter opts)))
          "you must supply batch-size and it must be an integer")
  (let [config (:multi-dataset-iter opts)
        {add-input :add-input
         add-input-hot :add-input-one-hot
         add-output :add-output
         add-output-hot :add-output-one-hot
         add-reader :add-reader
         add-seq-reader :add-seq-reader
         alignment :alignment-mode
         batch-size :batch-size} config
        {reader-name :reader-name
         first-column :first-column
         last-column :last-column} add-input
        {hot-reader-name :reader-name
         hot-column :column
         hot-num-classes :n-classes} add-input-hot
        {output-reader-name :reader-name
         output-first-column :first-column
         output-last-column :last-column} add-output
        {hot-output-reader-name :reader-name
         hot-output-column :column
         hot-output-n-classes :n-classes} add-output-hot
        {record-reader-name :reader-name
         rr :record-reader} add-reader
        {seq-reader-name :reader-name
         seq-rr :record-reader} add-seq-reader]
    (.build
     (as-> (RecordReaderMultiDataSetIterator$Builder. batch-size) b
      (cond (contains? config :add-input)
            (if (contains-many? add-input :first-column :last-column)
              (doto b (.addInput reader-name first-column last-column))
              (doto b (.addInput reader-name)))
            (and (contains? config :add-input-one-hot)
                 (contains-many? add-input-hot :reader-name :column :n-classes))
            (doto b (.addInputOneHot hot-reader-name hot-column hot-num-classes))
            (contains? config :add-output)
            (if (contains-many? add-output :first-column :last-column)
              (doto b (.addOutput output-reader-name output-first-column output-last-column))
              (doto b (.addOutput output-reader-name)))
            (and (contains? config :add-output-one-hot)
                 (contains-many? add-output-hot :column :n-classes :reader-name))
            (doto b (.addOutputOneHot hot-output-reader-name hot-output-column
                                      hot-output-n-classes))
            (and (contains? config :add-reader)
                 (contains-many? add-reader :reader-name :record-reader))
            (doto b (.addReader record-reader-name rr))
            (and (contains? config :add-seq-reader)
                 (contains-many? add-seq-reader :reader-name :record-reader))
            (doto b (.addSequenceReader seq-reader-name seq-rr))
            (contains? config :alignment-mode)
            (doto b (.sequenceAlignmentMode (value-of {:multi-alignment-mode alignment}))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns which hae documentation for properly calling multimethod
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-record-reader-dataset-iterator
  ;; spec this
  "creates a new record reader dataset iterator by calling its constructor
  with the supplied args.  args are:

  :record-reader (record-reader) a record reader, see datavec.api.records.readers
  :batch-size (int) the batch size
  :label-idx (int) the index of the labels in a dataset
  :n-possible-labels (int) the number of possible labels
  :label-idx-from (int) starting column for range of columns containing labels in the dataset
  :label-idx-to (int) ending column for range of columns containing labels in the dataset
  :regression? (boolean) are we dealing with a regression or classification problem
  :max-num-batches (int) the maximum number of batches the iterator should go through
  :writeable-converter (map) a writeable converter config, see TBD for options

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/datavec/RecordReaderDataSetIterator.html"
  [& {:keys [record-reader batch-size label-idx n-possible-labels
             label-idx-from label-idx-to regression? max-num-batches
             writeable-converter]
      :as opts}]
  (iterator {:rr-dataset-iter opts}))

(defn new-seq-record-reader-dataset-iterator
  ;; spec this
  "creates a new sequence record reader dataset iterator by calling its constructor
  with the supplied args.  args are:

  :record-reader (record-reader) a record reader, see datavec.api.records.readers
  :mini-batch-size (int) the mini batch size
  :n-possible-labels (int) the number of possible labels
  :label-idx (int) the index of the labels in a dataset
  :regression? (boolean) are we dealing with a regression or classification problem
  :labels-reader (record-reader) a record reader specificaly for labels, see datavec.api.records.readers
  :features-reader (record-reader) a record reader specificaly for features, see datavec.api.records.readers
  :alignment-mode (keyword), one of :equal-length, :align-start, :align-end
   -see https://deeplearning4j.org/doc/org/deeplearning4j/datasets/datavec/SequenceRecordReaderDataSetIterator.AlignmentMode.html

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/datavec/SequenceRecordReaderDataSetIterator.html"
  [& {:keys [record-reader mini-batch-size n-possible-labels label-idx regression?
             labels-reader features-reader alignment-mode]
      :as opts}]
  (iterator {:seq-rr-dataset-iter opts}))

(defn new-record-reader-multi-dataset-iterator
  ;; spec this
  "creates a new record reader multi dataset iterator by calling its builder with
  the supplied args.  args are:

  :alignment-mode (keyword),  one of :equal-length, :align-start, :align-end
   -see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/datavec/RecordReaderMultiDataSetIterator.AlignmentMode.html
  :batch-size (int), size of the batchs the iterator uses for a single run
  :add-seq-reader (record-reader), a sequence record reader
   -see: datavec.api.records.readers
  :add-reader (record-reader), a record reader
   -see: datavec.api.records.readers
  :add-output-one-hot (map) {:reader-name (str) :column (int) :n-classes (int)}
  :add-input-one-hot (map) {:reader-name (str) :column (int) :n-classes (int)}
  :add-input (map) {:reader-name (str) :first-column (int) :last-column (int)}
  :add-output (map) {:reader-name (str) :first-column (int) :last-column (int)}"
  [& {:keys [alignment-mode batch-size add-seq-reader
             add-reader add-output-one-hot add-output
             add-input-one-hot add-input]
      :as opts}]
  (iterator {:multi-dataset-iter opts}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; record reader interaction fns for only record reader and seq record reader
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn current-batch
  "return index of the current batch"
  [iter]
  (.batch iter))

(defn load-from-meta-data
  [iter meta-data]
  (.loadFromMetaData iter meta-data))

(defn remove-data
  [iter]
  (doto iter
    (.remove)))
