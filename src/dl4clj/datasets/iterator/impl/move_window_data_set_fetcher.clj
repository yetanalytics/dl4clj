(ns ^{:doc "Moving window data fetcher. Handles rotation of matrices in all directions to generate more examples.

see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/MovingWindowDataSetFetcher.html"}
    dl4clj.datasets.iterator.impl.move-window-data-set-fetcher
  (:import [org.deeplearning4j.datasets.iterator.impl MovingWindowDataSetFetcher]))

(defn new-moving-window-data-set-fetcher
  "Moving window data fetcher. Handles rotation of matrices in all directions
  to generate more examples.

  :data-set (DataSet), the dataset to rotate

  :window-rows (int), number of rows to rotate

  :window-columns (int), number of columns to rotate

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/MovingWindowDataSetFetcher.html"
  [& {:keys [data-set window-rows window-columns]}]
  (MovingWindowDataSetFetcher. data-set window-rows window-columns))

(defn fetch!
  "Fetches the next dataset."
  [& {:keys [ds-fetcher n-examples]}]
  (doto ds-fetcher (.fetch n-examples)))
