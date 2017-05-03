(ns dl4clj.datasets.iterator.impl.multi-data-set-iterator-adapter
  (:import [org.deeplearning4j.datasets.iterator.impl MultiDataSetIteratorAdapter]))

(defn new-multi-data-set-iterator-adapter
  "Iterator that adapts a DataSetIterator to a MultiDataSetIterator"
  [iter]
  (MultiDataSetIteratorAdapter. iter))
