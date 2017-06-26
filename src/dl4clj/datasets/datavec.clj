(ns dl4clj.datasets.datavec
  (:import [org.deeplearning4j.datasets DataSets]
           [org.deeplearning4j.datasets.datavec RecordReaderDataSetIterator
            RecordReaderMultiDataSetIterator$Builder RecordReaderMultiDataSetIterator
            SequenceRecordReaderDataSetIterator])
  (:require [dl4clj.constants :refer [value-of]]
            [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]
            [datavec.api.io :refer [new-double-writable-converter
                                    new-float-writable-converter
                                    new-label-writer-converter
                                    new-self-writable-converter]]
            [datavec.api.records.interface :refer [reset-rr!]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; build in datasets
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def iris-ds (DataSets/iris))

(def mnist-ds (DataSets/mnist))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; record reader interaction fns for only record reader and seq record reader
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn current-batch
  "return index of the current batch"
  [iter]
  (.batch iter))

(defn load-from-meta-data
  [& {:keys [iter meta-data]}]
  (.loadFromMetaData iter meta-data))

(defn remove-data!
  [iter]
  (doto iter
    (.remove)))
