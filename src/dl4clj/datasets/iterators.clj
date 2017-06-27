(ns dl4clj.datasets.iterators
  (:import [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator
            RecordReaderMultiDataSetIterator$Builder RecordReaderMultiDataSetIterator
            SequenceRecordReaderDataSetIterator]
           [org.deeplearning4j.datasets.iterator
            DoublesDataSetIterator FloatsDataSetIterator INDArrayDataSetIterator
            AsyncDataSetIterator AsyncMultiDataSetIterator CombinedPreProcessor
            CombinedPreProcessor$Builder CurvesDataSetIterator IteratorDataSetIterator
            IteratorMultiDataSetIterator MovingWindowBaseDataSetIterator
            MultipleEpochsIterator ReconstructionDataSetIterator SamplingDataSetIterator
            ExistingDataSetIterator]
           [org.deeplearning4j.datasets.iterator.impl MultiDataSetIteratorAdapter
            ListDataSetIterator SingletonMultiDataSetIterator]
           [org.deeplearning4j.datasets.iterator.impl
            CifarDataSetIterator IrisDataSetIterator LFWDataSetIterator
            MnistDataSetIterator RawMnistDataSetIterator]
           [java.util Random])
  (:require [dl4clj.constants :refer [value-of]]
            [dl4clj.berkeley :refer [new-pair]]
            [dl4clj.helpers :refer :all]
            [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]
            [datavec.api.records.interface :refer [reset-rr!]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;multimethod
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti iterator
  "Multimethod that builds a dataset iterator based on the supplied type and opts"
  generic-dispatching-fn)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; record reader dataset iterator mulimethods
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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
    (let [r (reset-rr! rr)]
     (if (contains? config :writeable-converter)
      (cond (contains-many? config :batch-size :label-idx-from :label-idx-to
                            :n-possible-labels :max-num-batches :regression?)
            (RecordReaderDataSetIterator.
             r
             converter
             batch-size l-idx-from l-idx-to n-labels
             max-n-batches regression?)
            (contains-many? config :batch-size :label-idx :n-possible-labels
                            :max-num-batches :regression?)
            (RecordReaderDataSetIterator.
             r
             converter
             batch-size label-idx n-labels max-n-batches regression?)
            (contains-many? config :batch-size :label-idx :n-possible-labels :regression?)
            (RecordReaderDataSetIterator.
             r
             converter
             batch-size label-idx n-labels regression?)
            (contains-many? :batch-size :label-idx :n-possible-labels)
            (RecordReaderDataSetIterator.
             r
             converter
             batch-size label-idx n-labels)
            (contains? config :batch-size)
            (RecordReaderDataSetIterator.
             r
             converter
             batch-size)
            :else
            (assert false "you must provide a record reader, writeable converter and a batch size"))
      (cond (contains-many? config :batch-size :label-idx :n-possible-labels :max-num-batches)
            (RecordReaderDataSetIterator.
             r batch-size label-idx n-labels max-n-batches)
            (contains-many? config :batch-size :label-idx-from :label-idx-to :regression?)
            (RecordReaderDataSetIterator.
             r batch-size l-idx-from l-idx-to regression?)
            (contains-many? config :batch-size :label-idx :n-possible-labels)
            (RecordReaderDataSetIterator.
             r batch-size label-idx n-labels)
            (contains? config :batch-size)
            (RecordReaderDataSetIterator.
             r batch-size)
            :else
            (assert false "you must supply a record reader and a batch size"))))))

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
    (let [r (reset-rr! rr)
          features-r (reset-rr! features-reader)
          labels-r (reset-rr! labels-reader)]
     (if (contains-many? config :labels-reader :features-reader)
      (cond (contains-many? config :mini-batch-size :n-possible-labels
                            :regression? :alignment-mode)
            (SequenceRecordReaderDataSetIterator.
             features-r labels-r
             m-batch-size n-labels regression?
             (value-of {:seq-alignment-mode alignment}))
            (contains-many? config :mini-batch-size :n-possible-labels :regression?)
            (SequenceRecordReaderDataSetIterator.
             features-r labels-r
             m-batch-size n-labels regression?)
            (contains-many? config :mini-batch-size :n-possible-labels)
            (SequenceRecordReaderDataSetIterator.
             features-r labels-r
             m-batch-size n-labels)
            :else
            (assert false "if youre supplying seperate labels and features readers,
you must supply atleast a batch size and the number of possible labels"))
      (cond (contains-many? config :mini-batch-size :n-possible-labels :label-idx
                            :regression?)
            (SequenceRecordReaderDataSetIterator.
             r m-batch-size n-labels label-idx regression?)
            (contains-many? config :mini-batch-size :n-possible-labels :label-idx)
            (SequenceRecordReaderDataSetIterator.
             r m-batch-size n-labels label-idx)
            :else
            (assert false "if you're supplying a single record reader for the features and the labels,
you need to suply atleast the mini batch size, number of possible labels and the column index of the labels"))))))

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
     (let [b (RecordReaderMultiDataSetIterator$Builder. batch-size)
           r (reset-rr! rr)
           seq-r (reset-rr! seq-rr)]
       (cond-> b
         (and (contains? config :add-reader)
              (contains-many? add-reader :reader-name :record-reader))
         (doto (.addReader record-reader-name r))
         (and (contains? config :add-seq-reader)
              (contains-many? add-seq-reader :reader-name :record-reader))
         (doto (.addSequenceReader seq-reader-name seq-r))
         (and (contains? config :add-input)
              (contains-many? add-input :first-column :last-column))
         (doto (.addInput reader-name first-column last-column))
         (and (contains? config :add-input-one-hot)
              (contains-many? add-input-hot :reader-name :column :n-classes))
         (doto (.addInputOneHot hot-reader-name hot-column hot-num-classes))
         (and (contains? config :add-output)
              (contains-many? add-output :first-column :last-column))
         (doto (.addOutput output-reader-name output-first-column output-last-column))
         (and (contains? config :add-output-one-hot)
              (contains-many? add-output-hot :column :n-classes :reader-name))
         (doto (.addOutputOneHot hot-output-reader-name hot-output-column
                                   hot-output-n-classes))
         (contains? config :alignment-mode)
         (doto (.sequenceAlignmentMode (value-of {:multi-alignment-mode alignment}))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dataset iterator mulimethods
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmethod iterator :doubles-dataset-iterator [opts]
  (let [config (:doubles-dataset-iter opts)
        {features :features
         labels :labels
         batch-size :batch-size} config]
    (DoublesDataSetIterator. [(new-pair :p1 (double-array features)
                                        :p2 (double-array labels))]
                             batch-size)))

(defmethod iterator :floats-dataset-iter [opts]
  (let [config (:floats-dataset-iter opts)
        {features :features
         labels :labels
         batch-size :batch-size} config]
    (FloatsDataSetIterator. [(new-pair :p1 (float-array features)
                                       :p2 (float-array labels))]
                            batch-size)))

(defmethod iterator :INDArray-dataset-iter [opts]
  (let [config (:INDArray-dataset-iter opts)
        {features :features
         labels :labels
         batch-size :batch-size} config]
    (INDArrayDataSetIterator. [(new-pair :p1 features :p2 labels)] batch-size)))

(defmethod iterator :iterator-multi-dataset-iter [opts]
  (let [config (:iterator-multi-dataset-iter opts)
        {iter :multi-dataset-iter
         batch-size :batch-size} config]
    (IteratorMultiDataSetIterator. (reset-if-not-at-start! iter) batch-size)))

(defmethod iterator :iterator-dataset-iter [opts]
  (let [config (:iterator-dataset-iter opts)
        {iter :iter
         batch-size :batch-size} config]
    (IteratorDataSetIterator. (reset-if-not-at-start! iter) batch-size)))

(defmethod iterator :async-multi-dataset-iter [opts]
  (let [config (:async-multi-dataset-iter opts)
        {iter :multi-dataset-iter
         que-l :que-length} config]
    (AsyncMultiDataSetIterator. (reset-if-not-at-start! iter) que-l)))

(defmethod iterator :moving-window-base-dataset-iter [opts]
  (let [config (:moving-window-base-dataset-iter opts)
        {batch :batch-size
         n-examples :n-examples
         data :dataset
         window-rows :window-rows
         window-columns :window-columns} config]
    (MovingWindowBaseDataSetIterator. batch n-examples data window-rows window-columns)))

(defmethod iterator :multiple-epochs-iter [opts]
  (let [config (:multiple-epochs-iter opts)
        {iter :iter
         q-size :que-size
         t-iterations :total-iterations
         n-epochs :n-epochs
         ds :dataset} config]
    (if (contains? config :n-epochs)
      (cond (contains-many? config :iter :que-size)
            (MultipleEpochsIterator. n-epochs (reset-if-not-at-start! iter) q-size)
            (contains? config :iter)
            (MultipleEpochsIterator. n-epochs (reset-if-not-at-start! iter))
            (contains? config :dataset)
            (MultipleEpochsIterator. n-epochs ds)
            :else
            (assert false "if you provide the number of epochs, you also need to provide either an iterator or a dataset"))
      (cond (contains-many? config :iter :que-size :total-iterations)
            (MultipleEpochsIterator. (reset-if-not-at-start! iter) q-size t-iterations)
            (contains-many? config :iter :total-iterations)
            (MultipleEpochsIterator. (reset-if-not-at-start! iter) t-iterations)
            :else
            (assert false "if you dont supply the number of epochs, you must supply atleast a dataset iterator and the total number of iterations")))))

(defmethod iterator :reconstruction-dataset-iter [opts]
  (let [config (:reconstruction-dataset-iter opts)
        iter (:iter config)]
    (ReconstructionDataSetIterator. (reset-if-not-at-start! iter))))

(defmethod iterator :sampling-dataset-iter [opts]
  (let [config (:sampling-dataset-iter opts)
        {ds :sampling-source
         batch-size :batch-size
         n-samples :total-n-samples} config]
    (SamplingDataSetIterator. ds batch-size n-samples)))

(defmethod iterator :existing-dataset-iter [opts]
  (let [config (:existing-dataset-iter opts)
        {iterable :dataset
         n-examples :total-examples
         n-features :n-features
         n-labels :n-labels
         labels :labels
         ds-iter :iter} config]
    (assert (or (contains? config :dataset)
                (contains? config :iter))
            "you must supply a dataset or a dataset iterator")
    (if (contains? config :iter)
      (if (contains? config :labels)
        (ExistingDataSetIterator. (reset-if-not-at-start! ds-iter) labels)
        (ExistingDataSetIterator. (reset-if-not-at-start! ds-iter)))
      (cond (contains-many? config :dataset :total-examples :n-features :n-labels)
            (ExistingDataSetIterator. iterable n-examples n-features n-labels)
            (contains-many? config :dataset :labels)
            (ExistingDataSetIterator. iterable labels)
            :else
            (ExistingDataSetIterator. iterable)))))

(defmethod iterator :async-dataset-iter [opts]
  (let [config (:async-dataset-iter opts)
        {ds-iter :iter
         que-size :que-size
         que :que} config]
    (let [i (reset-if-not-at-start! ds-iter)]
      (cond (contains-many? config :que :que-size :iter)
            (AsyncDataSetIterator. i que-size que)
            (contains-many? config :iter :que-size)
            (AsyncDataSetIterator. i que-size)
            (contains? config :iter)
            (AsyncDataSetIterator. i)
            :else
            (assert false "you must atleast provide a dataset iterator")))))

(defmethod iterator :ds-iter-to-multi-ds-iter [opts]
  (let [iter (reset-if-not-at-start! (:iter (:ds-iter-to-multi-ds-iter opts)))]
    (MultiDataSetIteratorAdapter. iter)))

(defmethod iterator :list-ds-iter [opts]
  (let [conf (:list-ds-iter opts)
        {ds :dataset
         batch-size :batch-size} conf]
    (if (contains? conf :batch-size)
      (ListDataSetIterator. ds batch-size)
      (ListDataSetIterator. ds))))

(defmethod iterator :multi-ds-to-multi-ds-iter [opts]
  (let [mds (:multi-dataset (:multi-ds-to-multi-ds-iter opts))]
    (SingletonMultiDataSetIterator. mds)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; default dataset iterator mulimethods
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmethod iterator :curves-dataset-iter [opts]
  (let [config (:curves-dataset-iter opts)
        {batch :batch-size
         n-examples :n-examples} config]
    (CurvesDataSetIterator. batch n-examples)))

(defmethod iterator :cifar-dataset-iter [opts]
  (let [config (:cifar-dataset-iter opts)
        {batch-size :batch-size
         n-examples :n-examples
         img-dims :img-dimgs
         train? :train?
         use-special-pre-process-cifar? :use-special-pre-process-cifar?
         n-possible-labels :n-possible-labels
         img-transform :img-transform} config
        img (int-array img-dims)]
    (cond (contains-many? config :batch-size :n-examples :img-dims :n-possible-labels
                        :img-transform :use-special-pre-process-cifar? :train?)
        (CifarDataSetIterator. batch-size n-examples img n-possible-labels
                               img-transform use-special-pre-process-cifar? train?)
        (contains-many? config :batch-size :n-examples :img-dims :use-special-pre-process-cifar? :train?)
        (CifarDataSetIterator. batch-size n-examples img use-special-pre-process-cifar? train?)
        (contains-many? config :batch-size :n-examples :img-dims :train?)
        (CifarDataSetIterator. batch-size n-examples img train?)
        (contains-many? config :batch-size :n-examples :img-dims)
        (CifarDataSetIterator. batch-size n-examples img)
        (contains-many? config :batch-size :n-examples :train?)
        (CifarDataSetIterator. batch-size n-examples train?)
        (contains-many? config :batch-size :img-dims)
        (CifarDataSetIterator. batch-size img)
        (contains-many? config :batch-size :n-examples)
        (CifarDataSetIterator. batch-size n-examples)
        :else
        (assert (and (contains? config :batch-size)
                     (or (contains? config :img-dims)
                         (contains? config :n-examples)))
                "you must provide atleast a batch size and number of examples or a batch size and the desired demensions of the images"))))

(defmethod iterator :iris-dataset-iter [opts]
  (let [config (:iris-dataset-iter opts)
        {batch-size :batch-size
         n-examples :n-examples} config]
    (IrisDataSetIterator. batch-size n-examples)))

(defmethod iterator :lfw-dataset-iter [opts]
  (let [config (:lfw-dataset-iter opts)
        {img-dims :img-dims
         batch-size :batch-size
         n-examples :n-examples
         use-subset? :use-subset?
         train? :train?
         split-train-test :split-train-test
         n-labels :n-labels
         seed :seed
         label-generator :label-generator
         image-transform :image-transform} config
        img (int-array img-dims)
        rng (new Random seed)]
    (cond (contains-many? config :batch-size :n-examples :img-dims :n-labels :use-subset?
                        :label-generator :train? :split-train-test :rng :image-transform)
        (LFWDataSetIterator. batch-size n-examples img n-labels use-subset?
                             label-generator train? split-train-test image-transform
                             rng)
        (contains-many? config :batch-size :n-examples :img-dims :n-labels :use-subset?
                        :label-generator :train? :split-train-test :rng)
        (LFWDataSetIterator. batch-size n-examples img n-labels use-subset?
                             label-generator train? split-train-test rng)
        (contains-many? config :batch-size :n-examples :img-dims :n-labels :use-subset?
                        :train? :split-train-test :rng)
        (LFWDataSetIterator. batch-size n-examples img n-labels use-subset?
                             train? split-train-test rng)
        (contains-many? config :batch-size :n-examples :n-labels :train? :split-train-test)
        (LFWDataSetIterator. batch-size n-examples n-labels train? split-train-test)
        (contains-many? config :batch-size :n-examples :img-dims :train? :split-train-test)
        (LFWDataSetIterator. batch-size n-examples img train? split-train-test)
        (contains-many? config :batch-size :n-examples :img-dims)
        (LFWDataSetIterator. batch-size n-examples img)
        (contains-many? config :batch-size :img-dims :use-subset?)
        (LFWDataSetIterator. batch-size img use-subset?)
        (contains-many? config :batch-size :n-examples)
        (LFWDataSetIterator. batch-size n-examples)
        (contains? config :img-dims)
        (LFWDataSetIterator. img)
        :else
        (assert false "you must supply atleast the desired image dimensions for the data"))))

(defmethod iterator :mnist-dataset-iter [opts]
  (let [config (:mnist-dataset-iter opts)
        {batch-size :batch-size
         train? :train?
         seed :seed
         n-examples :n-examples
         binarize? :binarize?
         shuffle? :shuffle?
         batch :batch} config]
    (cond (contains-many? config :batch :n-examples :binarize? :train? :shuffle? :seed)
          (MnistDataSetIterator. batch n-examples binarize? train? shuffle? (long seed))
          (contains-many? config :batch-size :train? :seed)
          (MnistDataSetIterator. batch-size train? (int seed))
          (contains-many? config :batch :n-examples :binarize?)
          (MnistDataSetIterator. batch n-examples binarize?)
          (contains-many? config :batch :n-examples)
          (MnistDataSetIterator. batch n-examples)
          :else
          (assert false "you must atleast supply a batch and number of examples"))))

(defmethod iterator :raw-mnist-dataset-iter [opts]
  (let [config (:raw-mnist-dataset-iter opts)
        {batch :batch
         n-examples :n-examples} config]
    (RawMnistDataSetIterator. batch n-examples)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; record reader dataset iterators user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-record-reader-dataset-iterator
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

  :writeable-converter (writable), converts a writable to another data type
   - need to have a central source of their creation
     - https://deeplearning4j.org/datavecdoc/org/datavec/api/writable/Writable.html
     - the classes which implement this interface
   - opts are new-double-writable-converter, new-float-writable-converter
     new-label-writer-converter, new-self-writable-converter
   - see: dl4clj.datavec.api.io

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/datavec/RecordReaderDataSetIterator.html"
  [& {:keys [record-reader batch-size label-idx n-possible-labels
             label-idx-from label-idx-to regression? max-num-batches
             writeable-converter]
      :as opts}]
  (iterator {:rr-dataset-iter opts}))

(defn new-seq-record-reader-dataset-iterator
  "creates a new sequence record reader dataset iterator by calling its constructor
  with the supplied args.  args are:

  :record-reader (sequence-record-reader) a record reader, see datavec.api.records.readers

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

  :add-seq-reader (map) {:reader-name (str) :record-reader (record-reader)}, a sequence record reader
   -see: datavec.api.records.readers

  :add-reader (map) {:reader-name (str) :record-reader (record-reader)}, a record reader
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
;; dataset iterators user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-async-dataset-iterator
  "AsyncDataSetIterator takes an existing DataFmulSetIterator and loads one or more
  DataSet objects from it using a separate thread. For data sets where
  (next! some-iterator) is long running (limited by disk read or processing time for example)
  this may improve performance by loading the next DataSet asynchronously
  (i.e., while training is continuing on the previous DataSet).

  Obviously this may use additional memory.
  Note however that due to asynchronous loading of data, (next! iter n) is not supported.

  :iter (ds-iterator), a dataset iterator

  :que-size (int), the size of the que

  :que (blocking-que), the que containing the dataset

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/AsyncDataSetIterator.html"
  [& {:keys [iter que-size que]
      :as opts}]
  (iterator {:async-dataset-iter opts}))

(defn new-existing-dataset-iterator
  "This wrapper provides DataSetIterator interface to existing datasets or dataset iterators

  :dataset (iterable), an iterable object, some dataset

  :total-examples (int), the total number of examples

  :n-features (int), the total number of features in the dataset

  :n-labels (int), the number of labels in a dataset

  :labels (list), a list of labels as strings

  :iter (iterator), a dataset iterator

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/ExistingDataSetIterator.html"
  [& {:keys [dataset total-examples n-features n-labels labels iter]
      :as opts}]
  (iterator {:existing-dataset-iter opts}))

(defn new-sampling-dataset-iterator
  "A wrapper for a dataset to sample from.
  This will randomly sample from the given dataset.

  :sampling-source (dataset), the dataset to sample from

  :batch-size (int), the batch size

  :total-n-samples (int), the total number of desired samples from the dataset

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/SamplingDataSetIterator.html"
  [& {:keys [sampling-source batch-size total-n-samples]
      :as opts}]
  (iterator {:sampling-dataset-iter opts}))

(defn new-reconstruction-dataset-iterator
  "Wraps a dataset iterator, setting the first (feature matrix) as the labels.

  ds-iter (iterator), the iterator to wrap

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/ReconstructionDataSetIterator.html"
  [& {:keys [iter]
      :as opts}]
  (iterator {:reconstruction-dataset-iter opts}))

(defn new-multiple-epochs-iterator
  "A dataset iterator for doing multiple passes over a dataset

  :iter (dataset iterator), an iterator for a dataset

  :que-size (int), the size for the multiple iterations (improve this desc)

  :total-iterations (long), the total number of times to run through the dataset

  :n-epochs (int), the number of epochs to run

  :dataset (dataset), a dataset

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/MultipleEpochsIterator.html"
  [& {:keys [iter que-size total-iterations n-epochs dataset]
      :as opts}]
  (iterator {:multiple-epochs-iter opts}))

(defn new-moving-window-base-dataset-iterator
  ;; currently can't test this until I figure out the issue im running into with
  ;; moving window dataset fetcher
  ;; this calls the fetcher behind the scene
  "DataSetIterator for moving window (rotating matrices)

  :batch-size (int), the batch size

  :n-examples (int), the total number of examples

  :dataset (dataset), a dataset to make new examples from

  :window-rows (int), the number of rows to rotate

  :window-columns (int), the number of columns to rotate

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/MovingWindowBaseDataSetIterator.html"
  [& {:keys [batch-size n-examples dataset window-rows window-columns]
      :as opts}]
  (iterator {:moving-window-base-dataset-iter opts}))

(defn new-async-multi-dataset-iterator
  "Async prefetching iterator wrapper for MultiDataSetIterator implementations

  use caution when using this with a CUDA backend

  :multi-dataset-iter (multidataset iterator), iterator to wrap

  :que-length (int), length of the que for async processing

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/AsyncMultiDataSetIterator.html"
  [& {:keys [multi-dataset-iter que-length]
      :as opts}]
  (iterator {:async-multi-dataset-iter opts}))

(defn new-iterator-dataset-iterator
  "A DataSetIterator that works on an Iterator, combining and splitting the input
  DataSet objects as required to get a consistent batch size.

  Typically used in Spark training, but may be used elsewhere.
  NOTE: reset method is not supported here.

  :iter (iter), an iterator containing datasets

  :batch-size (int), the batch size

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/IteratorDataSetIterator.html"
  [& {:keys [iter batch-size]
      :as opts}]
  (iterator {:iterator-dataset-iter opts}))

(defn new-iterator-multi-dataset-iterator
  "A DataSetIterator that works on an Iterator, combining and splitting the input
  DataSet objects as required to get a consistent batch size.

  Typically used in Spark training, but may be used elsewhere.
  NOTE: reset method is not supported here.

  :multi-dataset-iter (iter) an iterator containing multiple datasets

  :batch-size (int), the batch size

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/IteratorMultiDataSetIterator.html"
  [& {:keys [multi-dataset-iter batch-size]
      :as opts}]
  (iterator {:iterator-multi-dataset-iter opts}))

(defn new-doubles-dataset-iterator
  "creates a dataset iterator which iterates over the supplied features and labels

  :features (coll of doubles), a collection of doubles which acts as inputs
   - [0.2 0.4 ...]

  :labels (coll of doubles), a collection of doubles which acts as targets
   - [0.4 0.8 ...]

  :batch-size (int), the batch size

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/DoublesDataSetIterator.html"
  [& {:keys [features labels batch-size]
      :as opts}]
  (iterator {:doubles-dataset-iter opts}))

(defn new-floats-dataset-iterator
  "creates a dataset iterator which iterates over the supplied iterable

  :features (coll of floats), a collection of floats which acts as inputs

  :labels (coll of floats), a collection of floats which acts as the targets

  :batch-size (int), the batch size

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/FloatsDataSetIterator.html"
  [& {:keys [features labels batch-size]
      :as opts}]
  (iterator {:floats-dataset-iter opts}))

(defn new-INDArray-dataset-iterator
  "creates a dataset iterator given a pair of INDArrays and a batch-size

  :features (INDArray), an INDArray which acts as inputs
   - see: nd4clj.linalg.factory.nd4j

  :labels (INDArray), an INDArray which as the targets
   - see: nd4clj.linalg.factory.nd4j

  :batch-size (int), the batch size

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/INDArrayDataSetIterator.html"
  [& {:keys [iterable batch-size]
      :as opts}]
  (iterator {:INDArray-dataset-iter opts}))

(defn new-multi-data-set-iterator-adapter
  "Iterator that adapts a DataSetIterator to a MultiDataSetIterator"
  [& {:keys [iter]
      :as opts}]
  (iterator {:ds-iter-to-multi-ds-iter opts}))

(defn new-list-dataset-iterator
  "creates a new list data set iterator given a collection of datasets.

  :dataset (collection), a collection of dataset examples
   - from (as-list dataset), from nd4clj.linalg.dataset.api.data-set

  :batch-size (int), the batch size, if not supplied, defaults to 5

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/ListDataSetIterator.html"
  [& {:keys [dataset batch-size]
      :as opts}]
  (iterator {:list-ds-iter opts}))

(defn new-singleton-multi-dataset-iterator
  "A very simple adapter class for converting a single MultiDataSet to a MultiDataSetIterator.

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/SingletonMultiDataSetIterator.html"
  [& {:keys [multi-dataset]
      :as opts}]
  (iterator {:multi-ds-to-multi-ds-iter opts}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; default dataset iterators user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-curves-dataset-iterator
  "creates a dataset iterator for curbes data

  :batch-size (int), the size of the batch

  :n-examples (int), the total number of examples

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/CurvesDataSetIterator.html"
  [& {:keys [batch-size n-examples]
      :as opts}]
  (iterator {:curves-dataset-iter opts}))

(defn new-cifar-data-set-iterator
  "Load the images from the cifar dataset,

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/CifarDataSetIterator.html
  and: https://github.com/szagoruyko/cifar.torch

  :batch-size (int), the batch size

  :n-examples (int), the number of examples from the ds to include in the iterator

  :img-dim (vector), desired dimensions of the images
   - should contain 3 ints

  :train? (boolean), are we training or testing?

  :use-special-pre-process-cifar? (boolean), are we going to use the predefined preprocessor built for this dataset
   - There is a special preProcessor used to normalize the dataset based on Sergey Zagoruyko example https://github.com/szagoruyko/cifar.torch

  :img-transform (map) config map for an image-transformation (as of writing this doc string, not implemented)

  :n-possible-labels (int), specify the number of possible outputs/tags/classes for a given image"
  [& {:keys [batch-size n-examples img-dims train?
             use-special-pre-process-cifar?
             n-possible-labels img-transform]
      :as opts}]
  (iterator {:cifar-dataset-iter opts}))

(defn new-iris-data-set-iterator
  "IrisDataSetIterator handles traversing through the Iris Data Set.

  :batch-size (int), size of the batch

  :n-examples (int), number of examples to iterator over

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/IrisDataSetIterator.html"
  [& {:keys [batch-size n-examples]
      :as opts}]
  (iterator {:iris-dataset-iter opts}))

(defn new-lfw-data-set-iterator
  "Creates a dataset iterator for the LFW image dataset.

  :img-dims (int-array), desired dimensions of the images

  :batch-size (int), the batch size

  :n-examples (int), number of examples to take from the dataset

  :use-subset? (boolean), use a subset of the dataset or the whole thing

  :train? (boolean, are we training a net or testing it

  :split-train-test (double), the division between training and testing datasets

  :n-labels (int), the number of possible classifications for a single image

  :seed (int), number used to keep randomization consistent

  :label-generator (label generator), call (new-parent-path-label-generator) or
   (new-pattern-path-label-generator opts) to create a label generator to use

  :image-transform (map), a transform to apply to the images,
   - as of writing this doc string, this functionality not implemented

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/LFWDataSetIterator.html"
  [& {:keys [img-dims batch-size n-examples use-subset? train? split-train-test
             n-labels seed label-generator image-transform]
      :as opts}]
  (iterator {:lfw-dataset-iter opts}))

(defn new-mnist-data-set-iterator
  "creates a dataset iterator for the Mnist dataset

  :batch-size (int), the batch size

  :train? (boolean), training or testing

  :seed (int), used to consistently randomize the dataset

  :n-examples (int), the overall number of examples

  :binarize? (boolean), whether to binarize mnist or not

  :shuffle? (boolean), whether to shuffle the dataset or not

  :batch (int), size of each patch
  - supplying batch-size will retrieve the entire dataset where as batch will get a subset

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/MnistDataSetIterator.html"
  [& {:keys [batch-size train? seed n-examples binarize? shuffle? rng-seed batch]
      :as opts}]
  (iterator {:mnist-dataset-iter opts}))

(defn new-raw-mnist-data-set-iterator
  "Mnist data with scaled pixels

  :batch (int) size of each patch

  :n-examples (int), the overall number of examples

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/RawMnistDataSetIterator.html"
  [& {:keys [batch n-examples]
      :as opts}]
  (iterator {:raw-mnist-dataset-iter opts}))
