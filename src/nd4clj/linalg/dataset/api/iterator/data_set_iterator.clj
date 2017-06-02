(ns ^{:doc "see http://nd4j.org/doc/org/nd4j/linalg/dataset/api/iterator/DataSetIterator.html"}
  nd4clj.linalg.dataset.api.iterator.data-set-iterator
  (:refer-clojure :exclude [next])
  (:import [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.nd4j.linalg.dataset.api DataSetPreProcessor])) ;; must have changed

;; TODO
;; double check these fns are up to date
;; THIS CAN BE DELTED

(defn batch
  "Batch size"
  [^DataSetIterator this]
  (.batch this))

(defn cursor
  "The current cursor if applicable"
  [^DataSetIterator this]
  (.cursor this))

(defn input-columns
  "Input columns for the dataset"
  [^DataSetIterator this]
  (.inputColumns this))

(defn has-next
  "Get dataset iterator record reader labels"
  [^DataSetIterator this]
  (.hasNext this))

(defn next
  "Like the standard next method but allows a customizable number of examples returned"
  ([^DataSetIterator this]
   (.next this))
  ([^DataSetIterator this num]
   (.next this (int num))))

(defn num-examples
  "Total number of examples in the dataset"
  [^DataSetIterator this]
  (.numExamples this))

(defn reset
  "Resets the iterator back to the beginning"
  [^DataSetIterator this]
  (doto this (.reset)))

(defn set-pre-processor
  "Set a pre processor"
  [^DataSetIterator this ^DataSetPreProcessor pre-processor]
  (.setPreProcessor this pre-processor))

(defn total-examples
  "Total examples in the iterator"
  [^DataSetIterator this]
  (.reset this))

(defn total-outcomes
  "The number of labels for the dataset"
  [^DataSetIterator this]
  (.totalOutcomes this))
