(ns dl4clj.datasets.fetchers.default-dataset-fetchers
  (:import [org.deeplearning4j.datasets.fetchers
            MnistDataFetcher IrisDataFetcher CurvesDataFetcher])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn curves-fetcher
  "gets the Curves dataset."
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

  :seed (long) seed used for shuffling, shuffles unique per seed"
  [& {:keys [binarize? train? shuffle? seed]
      :as opts}]
  (cond (contains-many? opts :binarize? :train? :shuffle? :rng-seed)
        (MnistDataFetcher. binarize? train? shuffle? seed)
        (contains? opts :binarize)
        (MnistDataFetcher. binarize?)
        :else
        (MnistDataFetcher.)))
