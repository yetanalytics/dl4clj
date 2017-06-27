(ns dl4clj.datasets.api.fetchers
  (:import [org.deeplearning4j.datasets.fetchers BaseDataFetcher]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; api fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn fetcher-cursor
  "Direct access to a number represenative of iterating through a dataset"
  [fetcher]
  (.cursor fetcher))

(defn has-more?
  "returns true if there are more receords left to go through in a dataset"
  [fetcher]
  (.hasMore fetcher))

(defn input-column-length
  "The length of a feature vector for an individual example"
  [fetcher]
  (.inputColumns fetcher))

(defn reset-fetcher!
  "Returns the fetcher back to the beginning of the dataset, returns the fetcher"
  [fetcher]
  (doto fetcher (.reset)))

(defn n-examples-in-ds
  "The total number of examples"
  [fetcher]
  (.totalExamples fetcher))

(defn n-outcomes-in-ds
  "The number of labels for a dataset"
  [fetcher]
  (.totalOutcomes fetcher))
