(ns dl4clj.datasets.fetchers.default-dataset-fetchers
  (:import [org.deeplearning4j.datasets.fetchers
            MnistDataFetcher IrisDataFetcher CurvesDataFetcher])
  (:require [clojure.core.match :refer [match]]))

(defn curves-fetcher
  "gets the Curves dataset."
  [& {:keys [as-code?]
      :or {as-code? false}}]
  (if as-code?
    `(CurvesDataFetcher.)
    (CurvesDataFetcher.)))

(defn iris-fetcher
  "fetches the iris dataset"
  [& {:keys [as-code?]
      :or {as-code? false}}]
  (if as-code?
    `(IrisDataFetcher.)
    (IrisDataFetcher.)))

(defn mnist-fetcher
  "fetches the mnist dataset

  :binarize? (boolean) whether to binarize the dataset or not

  :train? (boolean) whether to the dataset is for training or testing

  :shuffle (boolean) whether to shulffle the dataset or not

  :seed (long) seed used for shuffling, shuffles unique per seed

  :as-code? (boolean), determines if the java object or the code for creating the object is returned"
  [& {:keys [binarize? train? shuffle? seed as-code?]
      :or {as-code? false}
      :as opts}]
  (let [code (match [opts]
                    [{:binarize? _ :train? _ :shuffle? _ :rng-seed _}]
                    `(MnistDataFetcher. ~binarize? ~train? ~shuffle? ~seed)
                    [{:binarize? _}]
                    `(MnistDataFetcher. ~binarize?)
                    :else
                    `(MnistDataFetcher.))]
    (if as-code?
      code
      (eval code))))
