(ns ^{:doc "Moving window data fetcher. Handles rotation of matrices in all directions to generate more examples.

see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/MovingWindowDataSetFetcher.html"}
    dl4clj.datasets.fetchers.move-window-data-set-fetcher
  (:import [org.deeplearning4j.datasets.iterator.impl MovingWindowDataSetFetcher])
  (:require [dl4clj.utils :refer [obj-or-code?]]))

;; can't get this working in tests
;; run into: java.lang.IllegalArgumentException: Only rotating matrices
;; could be due to a misunderstanding on my part
;; hence this namespace is not represented in datasets_test

;; this guy is called internally by MovingWindowBaseDataSetIterator

#_(defn new-moving-window-data-set-fetcher
  "Moving window data fetcher. Handles rotation of matrices in all directions
  to generate more examples.

  :data-set (DataSet), the dataset to rotate

  :window-rows (int), number of rows to rotate

  :window-columns (int), number of columns to rotate

  see: https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/impl/MovingWindowDataSetFetcher.html"
  [& {:keys [data-set window-rows window-columns as-code?]
      :or {as-code? true}}]
  (let [code `(MovingWindowDataSetFetcher. ~data-set ~window-rows ~window-columns)]
    (obj-or-code? as-code? code)))
