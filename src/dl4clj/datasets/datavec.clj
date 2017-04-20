(ns dl4clj.datasets.datavec
  (:import [org.deeplearning4j.datasets DataSets]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator
            RecordReaderMultiDataSetIterator$Builder RecordReaderMultiDataSetIterator
            SequenceRecordReaderDataSetIterator
            SequenceRecordReaderDataSetIterator$AlignmentMode]))

(def iris-ds (DataSets/iris))

(def mnist-ds (DataSets/mnist))

(defn iterator-type
  "dispatch fn for iterator"
  [opts]
  (first (keys opts)))

(defmulti iterator
  "Multimethod that builds a dataset iterator based on the supplied type and opts"
  iterator-type)

(defn alignment-type
  [k]
  (cond (= k :align-end)
        (SequenceRecordReaderDataSetIterator$AlignmentMode/ALIGN_END)
        (= k :align-start)
        (SequenceRecordReaderDataSetIterator$AlignmentMode/ALIGN_START)
        (= k :equal-length)
        (SequenceRecordReaderDataSetIterator$AlignmentMode/EQUAL_LENGTH)))

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
    (assert (= true (contains? config :record-reader)) "you must supply a record reader")
    (assert (= true (integer? batch-size)) "you must supply a batch-size")
    (if (contains? config :writeable-converter)
     (cond (and (integer? batch-size) (integer? l-idx-from)
               (integer? l-idx-to) (integer? n-labels) (integer? max-n-batches)
               (false? (nil? regression?)))
          (RecordReaderDataSetIterator. rr converter batch-size l-idx-from
                                        l-idx-to n-labels max-n-batches regression?)
          (and (integer? batch-size) (integer? label-idx)
               (integer? n-labels) (integer? max-n-batches) (false? (nil? regression?)))
          (RecordReaderDataSetIterator. rr converter batch-size label-idx n-labels
                                        max-n-batches regression?)
          (and (integer? batch-size) (integer? label-idx)
               (integer? n-labels) (false? (nil? regression?)))
          (RecordReaderDataSetIterator. rr converter batch-size label-idx
                                        n-labels regression?)
          (and (integer? batch-size) (integer? label-idx)
               (integer? n-labels))
          (RecordReaderDataSetIterator. rr converter batch-size label-idx n-labels)
          (integer? batch-size)
          (RecordReaderDataSetIterator. rr converter batch-size))
     (cond (and (integer? batch-size) (integer? label-idx) (integer? n-labels)
                (integer? max-n-batches))
           (RecordReaderDataSetIterator. rr batch-size label-idx n-labels max-n-batches)
           (and (integer? batch-size) (integer? l-idx-from) (integer? l-idx-to)
                (false? (nil? regression?)))
           (RecordReaderDataSetIterator. rr batch-size l-idx-from l-idx-to regression?)
           (and (integer? batch-size) (integer? label-idx) (integer? n-labels))
           (RecordReaderDataSetIterator. rr batch-size label-idx n-labels)
           :else
           (RecordReaderDataSetIterator. rr batch-size)))))

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
    (assert (= true (or (and (contains? config :labels-reader)
                             (contains? config :features-reader))
                        (contains? config :record-reader)))
            "you must supply a reader")
    (if (and (contains? config :labels-reader) (contains? config :features-reader))
      (cond (and (integer? m-batch-size) (integer? n-labels) (false? (nil? regression?))
                 (keyword? alignment))
            (SequenceRecordReaderDataSetIterator. features-reader labels-reader
                                                  m-batch-size n-labels regression?
                                                  (alignment-type alignment))
            (and (integer? m-batch-size) (integer? n-labels) (false? (nil? regression?)))
            (SequenceRecordReaderDataSetIterator. features-reader labels-reader
                                                  m-batch-size n-labels regression?)
            :else
            (SequenceRecordReaderDataSetIterator. features-reader labels-reader
                                                  m-batch-size n-labels))
      (cond (and (integer? m-batch-size) (integer? n-labels) (integer? label-idx)
                 (false? (nil? regression?)))
            (SequenceRecordReaderDataSetIterator. rr m-batch-size n-labels
                                                  label-idx regression?)
            :else
            (SequenceRecordReaderDataSetIterator. rr m-batch-size n-labels label-idx)))))

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
         seq-rr :record-reader} add-seq-reader
        b (RecordReaderMultiDataSetIterator$Builder. batch-size)]
    (.build
     (cond-> b
       (and (contains? config :add-input)
            (integer? first-column)
            (integer? last-column)
            (string? reader-name))
       (.addInput reader-name first-column last-column)
       (and (contains? config :add-input)
            (string? reader-name))
       (.addInput reader-name)
       (and (contains? config :add-input-one-hot)
            (string? hot-reader-name)
            (integer? hot-column)
            (integer? hot-num-classes))
       (.addInputOneHot hot-reader-name hot-column hot-num-classes)
       (and (contains? config :add-output)
            (string? output-reader-name)
            (integer? output-first-column)
            (integer? output-last-column))
       (.addOutput output-reader-name output-first-column output-last-column)
       (and (contains? config :add-output)
            (string? output-reader-name))
       (.addOutput output-reader-name)
       (and (contains? config :add-output-one-hot)
            (string? hot-output-reader-name)
            (integer? hot-output-column)
            (integer? hot-output-n-classes))
       (.addOutputOneHot hot-output-reader-name hot-output-column
                         hot-output-n-classes)
       (and (contains? config :add-reader)
            (string? record-reader-name)
            (contains? add-reader :record-reader))
       (.addReader record-reader-name rr)
       (and (contains? config :add-seq-reader)
            (string? seq-reader-name)
            (contains? add-seq-reader :record-reader))
       (.addSequenceReader seq-reader-name seq-rr)
       (and (contains? config :alignment-mode)
            (keyword? alignment))
       (.sequenceAlignmentMode (alignment-type alignment))))))

(defn async-supported?
  "is async supported?"
  [iter]
  (.asyncSupported iter))

(defn current-batch
  "return index of the current batch"
  [iter]
  (.batch iter))

(defn current-cursor
  [iter]
  (.cursor iter))

(defn get-labels
  [iter]
  (.getLabels iter))

(defn has-next?
  [iter]
  (.hasNext iter))

(defn n-input-columns
  [iter]
  (.inputColumns iter))

(defn load-from-meta-data
  [iter meta-data]
  (.loadFromMetaData iter meta-data))

(defn get-next
  ([iter]
   (.next iter))
  ([iter n]
   (.next iter n)))

(defn how-many-examples
  [iter]
  (.numExamples iter))

(defn remove-data
  [iter]
  (doto iter
    (.remove)))

(defn reset-iter
  [iter]
  (doto iter
    (.reset)))

(defn reset-supported?
  [iter]
  (.resetSupported iter))

(defn set-pre-processor
  [iter pre-processor]
  (doto iter
    (.setPreProcessor pre-processor)))

(defn total-examples
  [iter]
  (.totalExamples iter))

(defn total-outcomes
  [iter]
  (.totalOutcomes iter))
