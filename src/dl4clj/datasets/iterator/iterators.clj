(ns dl4clj.datasets.iterator.iterators
  (:import [org.deeplearning4j.datasets.iterator
            DoublesDataSetIterator FloatsDataSetIterator INDArrayDataSetIterator
            AsyncDataSetIterator AsyncMultiDataSetIterator CombinedPreProcessor
            CombinedPreProcessor$Builder CurvesDataSetIterator IteratorDataSetIterator
            IteratorMultiDataSetIterator MovingWindowBaseDataSetIterator
            MultipleEpochsIterator ReconstructionDataSetIterator SamplingDataSetIterator
            ExistingDataSetIterator]
           [org.deeplearning4j.datasets.iterator.impl MultiDataSetIteratorAdapter
            ListDataSetIterator SingletonMultiDataSetIterator])
  (:require [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]
            [dl4clj.berkeley :refer [new-pair]]
            [dl4clj.helpers :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method heavy lifting
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti iterators
  "multimethod for creating iterators"
  generic-dispatching-fn)

(defmethod iterators :doubles-dataset-iterator [opts]
  (let [config (:doubles-dataset-iterator opts)
        {features :features
         labels :labels
         batch-size :batch-size} config]
    (DoublesDataSetIterator. [(new-pair :p1 (double-array features)
                                        :p2 (double-array labels))]
                             batch-size)))

(defmethod iterators :floats-dataset-iterator [opts]
  (let [config (:floats-dataset-iterator opts)
        {features :features
         labels :labels
         batch-size :batch-size} config]
    (FloatsDataSetIterator. [(new-pair :p1 (float-array features)
                                      :p2 (float-array labels))]
                            batch-size)))

(defmethod iterators :INDArray-dataset-iterator [opts]
  (let [config (:INDArray-dataset-iterator opts)
        {features :features
         labels :labels
         batch-size :batch-size} config]
    (INDArrayDataSetIterator. [(new-pair :p1 features :p2 labels)] batch-size)))

(defmethod iterators :curves-dataset-iterator [opts]
  (let [config (:curves-dataset-iterator opts)
        {batch :batch-size
         n-examples :n-examples} config]
    (CurvesDataSetIterator. batch n-examples)))

(defmethod iterators :iterator-multi-dataset-iterator [opts]
  (let [config (:iterator-multi-dataset-iterator opts)
        {iter :multi-dataset-iter
         batch-size :batch-size} config]
    (IteratorMultiDataSetIterator. (reset-if-not-at-start! iter) batch-size)))

(defmethod iterators :iterator-dataset-iterator [opts]
  (let [config (:iterator-dataset-iterator opts)
        {iter :iter
         batch-size :batch-size} config]
    (IteratorDataSetIterator. (reset-if-not-at-start! iter) batch-size)))

(defmethod iterators :async-multi-dataset-iterator [opts]
  (let [config (:async-multi-dataset-iterator opts)
        {iter :multi-dataset-iter
         que-l :que-length} config]
    (AsyncMultiDataSetIterator. (reset-if-not-at-start! iter) que-l)))

(defmethod iterators :moving-window-base-dataset-iterator [opts]
  (let [config (:moving-window-base-dataset-iterator opts)
        {batch :batch-size
         n-examples :n-examples
         data :dataset
         window-rows :window-rows
         window-columns :window-columns} config]
    (MovingWindowBaseDataSetIterator. batch n-examples data window-rows window-columns)))

(defmethod iterators :multiple-epochs-iterator [opts]
  (let [config (:multiple-epochs-iterator opts)
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

(defmethod iterators :reconstruction-dataset-iterator [opts]
  (let [config (:reconstruction-dataset-iterator opts)
        iter (:iter config)]
    (ReconstructionDataSetIterator. (reset-if-not-at-start! iter))))

(defmethod iterators :sampling-dataset-iterator [opts]
  (let [config (:sampling-dataset-iterator opts)
        {ds :sampling-source
         batch-size :batch-size
         n-samples :total-n-samples} config]
    (SamplingDataSetIterator. ds batch-size n-samples)))

(defmethod iterators :existing-dataset-iterator [opts]
  (let [config (:existing-dataset-iterator opts)
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

(defmethod iterators :async-dataset-iterator [opts]
  (let [config (:async-dataset-iterator opts)
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

(defmethod iterators :ds-iter-to-multi-ds-iter [opts]
  (let [iter (reset-if-not-at-start! (:iter (:ds-iter-to-multi-ds-iter opts)))]
    (MultiDataSetIteratorAdapter. iter)))

(defmethod iterators :list-ds-iter [opts]
  (let [conf (:list-ds-iter opts)
        {ds :dataset
         batch-size :batch-size} conf]
    (if (contains? conf :batch-size)
      (ListDataSetIterator. ds batch-size)
      (ListDataSetIterator. ds))))

(defmethod iterators :multi-ds-to-multi-ds-iter [opts]
  (let [mds (:multi-dataset (:multi-ds-to-multi-ds-iter opts))]
    (SingletonMultiDataSetIterator. mds)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-combined-pre-processor
  "This is special preProcessor, that allows to combine multiple prerpocessors,
  and apply them to data sequentially.

   pre-processors (map), {(int) (pre-processor)
                         (int) (pre-processor)}

   - the keys are the desired indexes of the pre-processors (dataset index within the iterator)
   - values are the pre-processors themselves
   - pre-processors should be fit to a dataset or iterator before being combined

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/CombinedPreProcessor.html"
  [pre-processors]
  (loop [b (CombinedPreProcessor$Builder.)
         result pre-processors]
    (if (empty? result)
      (.build b)
      (let [data (first result)
            [idx pp] data]
        (recur (doto b (.addPreProcessor idx pp)) (rest result))))))

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
  (iterators {:async-dataset-iterator opts}))

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
  (iterators {:existing-dataset-iterator opts}))

(defn new-sampling-dataset-iterator
  "A wrapper for a dataset to sample from.
  This will randomly sample from the given dataset.

  :sampling-source (dataset), the dataset to sample from

  :batch-size (int), the batch size

  :total-n-samples (int), the total number of desired samples from the dataset

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/SamplingDataSetIterator.html"
  [& {:keys [sampling-source batch-size total-n-samples]
      :as opts}]
  (iterators {:sampling-dataset-iterator opts}))

(defn new-reconstruction-dataset-iterator
  "Wraps a dataset iterator, setting the first (feature matrix) as the labels.

  ds-iter (iterator), the iterator to wrap

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/ReconstructionDataSetIterator.html"
  [& {:keys [iter]
      :as opts}]
  (iterators {:reconstruction-dataset-iterator opts}))

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
  (iterators {:multiple-epochs-iterator opts}))


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
  (iterators {:moving-window-base-dataset-iterator opts}))

(defn new-async-multi-dataset-iterator
  "Async prefetching iterator wrapper for MultiDataSetIterator implementations

  use caution when using this with a CUDA backend

  :multi-dataset-iter (multidataset iterator), iterator to wrap

  :que-length (int), length of the que for async processing

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/AsyncMultiDataSetIterator.html"
  [& {:keys [multi-dataset-iter que-length]
      :as opts}]
  (iterators {:async-multi-dataset-iterator opts}))

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
  (iterators {:iterator-dataset-iterator opts}))

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
  (iterators {:iterator-multi-dataset-iterator opts}))

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
  (iterators {:doubles-dataset-iterator opts}))

(defn new-floats-dataset-iterator
  "creates a dataset iterator which iterates over the supplied iterable

  :features (coll of floats), a collection of floats which acts as inputs

  :labels (coll of floats), a collection of floats which acts as the targets

  :batch-size (int), the batch size

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/FloatsDataSetIterator.html"
  [& {:keys [features labels batch-size]
      :as opts}]
  (iterators {:floats-dataset-iterator opts}))

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
  (iterators {:INDArray-dataset-iterator opts}))

(defn new-curves-dataset-iterator
  "creates a dataset iterator for curbes data

  :batch-size (int), the size of the batch

  :n-examples (int), the total number of examples

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/CurvesDataSetIterator.html"
  [& {:keys [batch-size n-examples]
      :as opts}]
  (iterators {:curves-dataset-iterator opts}))

(defn new-multi-data-set-iterator-adapter
  "Iterator that adapts a DataSetIterator to a MultiDataSetIterator"
  [& {:keys [iter]
      :as opts}]
  (iterators {:ds-iter-to-multi-ds-iter opts}))

(defn new-list-dataset-iterator
  "creates a new list data set iterator given a collection of datasets.

  :dataset (collection), a collection of dataset examples
   - from (as-list dataset), from nd4clj.linalg.dataset.api.data-set

  :batch-size (int), the batch size, if not supplied, defaults to 5

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/ListDataSetIterator.html"
  [& {:keys [dataset batch-size]
      :as opts}]
  (iterators {:list-ds-iter opts}))

(defn new-singleton-multi-dataset-iterator
  "A very simple adapter class for converting a single MultiDataSet to a MultiDataSetIterator.

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/SingletonMultiDataSetIterator.html"
  [& {:keys [multi-dataset]
      :as opts}]
  (iterators {:multi-ds-to-multi-ds-iter opts}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; api calls
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn pre-process-iter-combined-pp!
  "Pre process a dataset sequentially using a combined pre-processor
   - the pre-processor is attached to the dataset"
  [& {:keys [dataset-iter dataset]}]
  (doto dataset-iter (.preProcess dataset)))
