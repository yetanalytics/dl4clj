(ns dl4clj.datasets.fetchers.default-dataset-fetchers
  (:import [org.deeplearning4j.datasets.fetchers
            MnistDataFetcher IrisDataFetcher CurvesDataFetcher])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn curves-fetcher
  "gets the Curves dataset.

  currently getten an EOF error on initialization
   -seems to be an issue with macs"
  []
  (CurvesDataFetcher.))

(defn iris-fetcher
  "fetches the iris dataset"
  []
  (IrisDataFetcher.))

(defn mnist-fetcher
  "fetches the mnist dataset

  :binarize? (boolean) whether to binarize the dataset or not
  :train? (boolean) whether to the dataset is for training or testing
  :shuffle (boolean) whether to shulffle the dataset or not
  :rng-seed (long) seed used for shuffling, shuffles unique per seed"
  [& {:keys [binarize? train? shuffle? rng-seed]
      :as opts}]
  (cond (contains-many? opts :binarize? :train? :shuffle? :rng-seed)
        (MnistDataFetcher. binarize? train? shuffle? rng-seed)
        (contains? opts :binarize)
        (MnistDataFetcher. binarize?)
        :else
        (MnistDataFetcher.)))

;; shared methods
;; fetch, hasMore, next
;; all the methods from base data fetcher

;; underlying mnist dataset stuff
;; https://deeplearning4j.org/doc/org/deeplearning4j/datasets/mnist/package-summary.html
;; don't think necessary to implement but come back to this
