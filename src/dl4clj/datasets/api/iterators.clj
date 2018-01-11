(ns dl4clj.datasets.api.iterators
  (:import [org.deeplearning4j.datasets.datavec
            RecordReaderDataSetIterator
            SequenceRecordReaderDataSetIterator]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code? eval-if-code]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; getters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn current-batch
  "return index of the current batch"
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.batch ~iter))
         :else
         (.batch iter)))

(defn get-batch-size
  "return the batch size"
  [iter & {:keys [as-code?]
      :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.batch ~iter))
         :else
         (.batch iter)))

(defn get-current-cursor
  "The current cursor if applicable"
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.cursor ~iter))
         :else
         (.cursor iter)))

(defn get-pre-processor
  "Returns preprocessors, if defined"
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getPreProcessor ~iter))
         :else
         (.getPreProcessor iter)))

(defn get-input-columns
  "Input columns for the dataset"
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.inputColumns ~iter))
         :else
         (.inputColumns iter)))

(defn next-n-examples!
  "Like the standard next method but allows a customizable number of examples returned"
  [& {:keys [iter n as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:iter (_ :guard seq?)
           :n (:or (_ :guard number?)
                   (_ :guard seq?))}]
         (obj-or-code? as-code? `(.next ~iter ~n))
         [{:iter _
           :n _}]
         (let [[iter-obj some-n] (eval-if-code [iter seq?]
                                               [n seq? number?])]
           (.next iter-obj some-n))))

(defn next-example!
  "returns the next example in an iterator"
  ;; TODO: has problems with multi-ds iterators, will need to figure this out
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.next ~iter))
         :else
         (.next iter)))

(defn get-num-examples
  "Total number of examples in the dataset"
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.numExamples ~iter))
         :else
         (.numExamples iter)))

(defn get-total-examples
  "Total examples in the iterator"
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.totalExamples ~iter))
         :else
         (.totalExamples iter)))

(defn get-total-outcomes
  "The number of labels for the dataset"
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.totalOutcomes ~iter))
         :else
         (.totalOutcomes iter)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn load-from-meta-data
  [& {:keys [iter meta-data as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:iter (_ :guard seq?)
           :meta-data (_ :guard seq?)}]
         (obj-or-code? as-code? `(.loadFromMetaData ~iter ~meta-data))
         :else
         (let [[iter-obj md-obj] (eval-if-code [iter seq?]
                                               [meta-data seq?])]
           (.loadFromMetaData iter-obj md-obj))))

(defn remove-data!
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~iter .remove))
         :else
         (doto iter .remove)))

(defn async-supported?
  "Does this DataSetIterator support asynchronous prefetching of multiple DataSet objects?
  Most DataSetIterators do, but in some cases it may not make sense to wrap
  this iterator in an iterator that does asynchronous prefetching."
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.asyncSupported ~iter))
         :else
         (.asyncSupported iter)))

(defn reset-supported?
  "Is resetting supported by this DataSetIterator?
  Many DataSetIterators do support resetting, but some don't"
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.resetSupported ~iter))
         :else
         (.resetSupported iter)))

(defn reset-iter!
  "Resets the iterator back to the beginning"
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~iter .reset))
         :else
         (doto iter .reset)))

(defn has-next?
  "checks to see if there is anymore data in the iterator"
  [iter & {:keys [as-code?]
           :or {as-code? true}}]
  (match [iter]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.hasNext ~iter))
         :else
         (.hasNext iter)))

(defn set-pre-processor!
  "Set a pre processor"
  [& {:keys [iter pre-processor as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:iter (_ :guard seq?)
           :pre-processor (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~iter (.setPreProcessor ~pre-processor)))
         :else
         (let [[iter-obj pp-obj] (eval-if-code [iter seq?] [pre-processor seq?])]
           (doto (if (has-next? iter-obj)
                   iter-obj
                   (reset-iter! iter-obj))
             (.setPreProcessor pp-obj)))))
