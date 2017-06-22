(ns ^{:doc "A DataSetIterator handles traversing through a dataset and preparing data for a nn
see: http://nd4j.org/doc/org/nd4j/linalg/dataset/api/iterator/DataSetIterator.html"}
    nd4clj.linalg.api.ds-iter
  (:import [org.nd4j.linalg.dataset.api.iterator DataSetIterator])
  (:require [dl4clj.datasets.iterator.iterators :refer [new-existing-dataset-iterator]]))

(defn async-supported?
  "Does this DataSetIterator support asynchronous prefetching of multiple DataSet objects?
  Most DataSetIterators do, but in some cases it may not make sense to wrap
  this iterator in an iterator that does asynchronous prefetching."
  [iter]
  (.asyncSupported iter))

(defn get-batch-size
  "return the batch size"
  [iter]
  (.batch iter))

(defn get-current-cursor
  "The current cursor if applicable"
  [iter]
  (.cursor iter))

(defn get-pre-processor
  "Returns preprocessors, if defined"
  [iter]
  (.getPreProcessor iter))

(defn get-input-columns
  "Input columns for the dataset"
  [iter]
  (.inputColumns iter))

(defn next-n-examples!
  "Like the standard next method but allows a customizable number of examples returned"
  [& {:keys [iter n]}]
  (.next iter n))

(defn next-example!
  "returns the next example in an iterator"
  [iter]
  (.next iter))

(defn get-num-examples
  "Total number of examples in the dataset"
  [iter]
  (.numExamples iter))

(defn reset-iter!
  "Resets the iterator back to the beginning"
  [iter]
  (doto iter (.reset)))

(defn reset-supported?
  "Is resetting supported by this DataSetIterator?
  Many DataSetIterators do support resetting, but some don't"
  [iter]
  (.resetSupported iter))

(defn set-pre-processor!
  "Set a pre processor"
  [& {:keys [iter pre-processor]}]
  (doto iter (.setPreProcessor pre-processor)))

(defn get-total-examples
  "Total examples in the iterator"
  [iter]
  (.totalExamples iter))

(defn get-total-outcomes
  "The number of labels for the dataset"
  [iter]
  (.totalOutcomes iter))

(defn has-next?
  "checks to see if there is anymore data in the iterator"
  [iter]
  (.hasNext iter))

(defn reset-if-empty?!
  "resets an iterator if we are at the end"
  [iter]
  (if (false? (has-next? iter))
    (reset-iter! iter)
    iter))

(defn reset-if-not-at-start!
  "checks the current cursor of the iterator and if not at 0 resets it"
  [iter]
  (if (not= 0 (get-current-cursor iter))
    (reset-iter! iter)
    iter))

(defn data-from-iter
  "returns all the data from an iterator as a lazy seq"
  [iter]
  (when (has-next? iter)
    (lazy-seq (cons (next-example! iter) (data-from-iter iter)))))

(defn iter-from-lazy-seq
  "creates an iterator for a lazy seq.  This iterator can be used in model training"
  [& {:keys [lazy-seq labels]
      :as opts}]
  (if (contains? opts :labels)
    (new-existing-dataset-iterator :iter (.iterator lazy-seq)
                                   :labels labels))
  (new-existing-dataset-iterator :iter (.iterator lazy-seq)))
