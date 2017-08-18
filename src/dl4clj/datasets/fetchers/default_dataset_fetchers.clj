(ns dl4clj.datasets.fetchers.default-dataset-fetchers
  (:import [org.deeplearning4j.datasets.fetchers
            MnistDataFetcher IrisDataFetcher CurvesDataFetcher])
  (:require [clojure.core.match :refer [match]]))

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
  (match [opts]
         [{:binarize? _ :train? _ :shuffle? _ :rng-seed _}]
         (MnistDataFetcher. binarize? train? shuffle? seed)
         [{:binarize? _}]
         (MnistDataFetcher. binarize?)
         :else
         (MnistDataFetcher.)))
