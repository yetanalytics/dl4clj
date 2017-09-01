(ns dl4clj.datasets.rearrange
  (:import [org.deeplearning4j.datasets.rearrange LocalUnstructuredDataFormatter])
  (:require [dl4clj.constants :as enum]
            [clojure.java.io :as io]))

;; this namespace is not reflected in the datasets_test namespace
(defn new-unstructured-formatter
  "Rearrange an unstructured dataset in to split test/train on the file system

  :destination-root-dir (str) file path to save rearranged dataset to

  :src-root-dir (str) file path to get the unstructured dataset from

  :labeling-type (keyword), one of :directory or :name

  :percent-train (double), the percentage of the dataset to use for training"
  [& {:keys [destination-root-dir src-root-dir labeling-type percent-train as-code?]
      :or {as-code? true}}]
  (let [code `(LocalUnstructuredDataFormatter. (io/as-file ~destination-root-dir)
                                   (io/as-file ~src-root-dir)
                                   (enum/value-of {:labeling-type ~labeling-type})
                                   ~percent-train)]))
