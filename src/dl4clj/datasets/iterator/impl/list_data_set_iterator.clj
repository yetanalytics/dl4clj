(ns ^{:doc "Wraps a data applyTransformToDestination collection
see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/ListDataSetIterator.html"}
    dl4clj.datasets.iterator.impl.list-data-set-iterator
  (:import [org.deeplearning4j.datasets.iterator.impl ListDataSetIterator]))

(defn new-list-data-set-iterator
  "creates a new list data set iterator given a dataset.

  :data-set (collection), a dataset in a clojure collection

  :batch (int), the batch size, if not supplied, defaults to 5

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/ListDataSetIterator.html"
  [& {:keys [data-set batch]
      :as opts}]
  (if (contains? opts :batch)
    (ListDataSetIterator. data-set batch)
    (ListDataSetIterator. data-set)))
