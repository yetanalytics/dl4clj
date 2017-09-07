(ns dl4clj.datasets.api.iterators
  (:import [org.deeplearning4j.datasets.datavec
            RecordReaderDataSetIterator
            SequenceRecordReaderDataSetIterator]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator])
  (:require [clojure.core.match :refer [match]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dataset iterators
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn current-batch
  "return index of the current batch"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.batch ~iter)
         :else
         (.batch iter)))

(defn get-batch-size
  "return the batch size"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.batch ~iter)
         :else
         (.batch iter)))

(defn load-from-meta-data
  [& {:keys [iter meta-data]
      :as opts}]
  (match [opts]
         [{:iter (_ :guard seq?)
           :meta-data (_ :guard seq?)}]
         `(.loadFromMetaData ~iter ~meta-data)
         :else
         (.loadFromMetaData iter meta-data)))

(defn remove-data!
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(doto ~iter .remove)
         :else
         (doto iter .remove)))

(defn async-supported?
  "Does this DataSetIterator support asynchronous prefetching of multiple DataSet objects?
  Most DataSetIterators do, but in some cases it may not make sense to wrap
  this iterator in an iterator that does asynchronous prefetching."
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.asyncSupported ~iter)
         :else
         (.asyncSupported iter)))

(defn reset-supported?
  "Is resetting supported by this DataSetIterator?
  Many DataSetIterators do support resetting, but some don't"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.resetSupported ~iter)
         :else
         (.resetSupported iter)))

(defn get-current-cursor
  "The current cursor if applicable"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.cursor ~iter)
         :else
         (.cursor iter)))

(defn get-pre-processor
  "Returns preprocessors, if defined"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.getPreProcessor ~iter)
         :else
         (.getPreProcessor iter)))

(defn get-input-columns
  "Input columns for the dataset"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.inputColumns ~iter)
         :else
         (.inputColumns iter)))

(defn next-n-examples!
  "Like the standard next method but allows a customizable number of examples returned"
  [& {:keys [iter n]
      :as opts}]
  (match [opts]
         [{:iter (_ :guard seq?)
           :n (:or (_ :guard number?)
                   (_ :guard seq?))}]
         `(.next ~iter ~n)
         [{:iter _
           :n _}]
         (.next iter n)))

(defn next-example!
  "returns the next example in an iterator"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.next ~iter)
         :else
         (.next iter)))

(defn get-num-examples
  "Total number of examples in the dataset"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.numExamples ~iter)
         :else
         (.numExamples iter)))

(defn reset-iter!
  "Resets the iterator back to the beginning"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(doto ~iter .reset)
         :else
         (doto iter .reset)))

(defn has-next?
  "checks to see if there is anymore data in the iterator"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.hasNext ~iter)
         :else
         (.hasNext iter)))

(defn set-pre-processor!
  "Set a pre processor"
  [& {:keys [iter pre-processor]
      :as opts}]
  (match [opts]
         [{:iter (_ :guard seq?)
           :pre-processor (_ :guard seq?)}]
         `(doto ~iter (.setPreProcessor ~pre-processor))
         :else
         (doto (if (has-next? iter)
                 iter
                 (reset-iter! iter))
           (.setPreProcessor pre-processor))))

(defn get-total-examples
  "Total examples in the iterator"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.totalExamples ~iter)
         :else
         (.totalExamples iter)))

(defn get-total-outcomes
  "The number of labels for the dataset"
  [iter]
  (match [iter]
         [(_ :guard seq?)]
         `(.totalOutcomes ~iter)
         :else
         (.totalOutcomes iter)))
