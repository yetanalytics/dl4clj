(ns ^{:doc "A DataSetIterator handles traversing through a dataset and preparing data for a nn
see: http://nd4j.org/doc/org/nd4j/linalg/dataset/api/iterator/DataSetIterator.html"}
    nd4clj.linalg.api.ds-iter
  (:import [org.nd4j.linalg.dataset.api.iterator DataSetIterator]))

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

(defn get-labels-rr
  "Get dataset iterator record reader labels"
  [iter]
  (.getLabels iter))

(defn get-pre-processor
  "Returns preprocessors, if defined"
  [iter]
  (.getPreProcessor iter))

(defn get-input-columns
  "Input columns for the dataset"
  [iter]
  (.inputColumns iter))

(defn next-n-examples
  "Like the standard next method but allows a customizable number of examples returned"
  [& {:keys [iter n]}]
  (.next iter n))

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
