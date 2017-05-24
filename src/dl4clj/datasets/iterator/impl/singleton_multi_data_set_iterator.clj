(ns ^{:doc "A very simple adapter class for converting a single MultiDataSet to a MultiDataSetIterator.

see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/SingletonMultiDataSetIterator.html"}
    dl4clj.datasets.iterator.impl.singleton-multi-data-set-iterator
  (:import [org.deeplearning4j.datasets.iterator.impl SingletonMultiDataSetIterator]))

(defn new-singleton-multi-data-set-iterator
  "A very simple adapter class for converting a single MultiDataSet to a MultiDataSetIterator.
  Returns a single MultiDataSet"
  [multi-data-set]
  (SingletonMultiDataSetIterator. multi-data-set))
