(ns dl4clj.datasets.fetchers.base-data-fetcher
  (:import [org.deeplearning4j.datasets.fetchers BaseDataFetcher]))

(defn cursor
  "Direct access to a number represenative of iterating through a dataset"
  [dataset]
  (.cursor dataset))

(defn get-label-name-at-idx
  "returns the name of a label at a specified idx

  :dataset a dataset
  :idx (int), index of the dataset you want the label for"
  [& {:keys [dataset idx]}]
  (.getLabelName dataset idx))

(defn has-more?
  "returns true if there are more receords left to go through in a dataset"
  [dataset]
  (.hasMore dataset))

(defn input-column-length
  "The length of a feature vector for an individual example"
  [dataset]
  (.inputColumns dataset))

;; also has fns known to be implemented else where
;; next, reset, setLabelNames, totalExamples, totalOutcomes
