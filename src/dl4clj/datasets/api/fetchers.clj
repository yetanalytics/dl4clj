(ns dl4clj.datasets.api.fetchers
  (:import [org.deeplearning4j.datasets.fetchers BaseDataFetcher])
  (:require [clojure.core.match :refer [match]]))

(defn fetcher-cursor
  "Direct access to a number represenative of iterating through a dataset"
  [fetcher]
  (match [fetcher]
         [(_ :guard seq?)]
         `(.cursor ~fetcher)
         :else
         (.cursor fetcher)))

(defn has-more?
  "returns true if there are more receords left to go through in a dataset"
  [fetcher]
  (match [fetcher]
         [(_ :guard seq?)]
         `(.hasMore ~fetcher)
         :else
         (.hasMore fetcher)))

(defn input-column-length
  "The length of a feature vector for an individual example"
  [fetcher]
  (match [fetcher]
         [(_ :guard seq?)]
         `(.inputColumns ~fetcher)
         :else
         (.inputColumns fetcher)))

(defn reset-fetcher!
  "Returns the fetcher back to the beginning of the dataset, returns the fetcher"
  [fetcher]
  (match [fetcher]
         [(_ :guard seq?)]
         `(doto ~fetcher .reset)
         :else
         (doto fetcher .reset)))

(defn n-examples-in-ds
  "The total number of examples"
  [fetcher]
  (match [fetcher]
         [(_ :guard seq?)]
         `(.totalExamples ~fetcher)
         :else
         (.totalExamples fetcher)))

(defn n-outcomes-in-ds
  "The number of labels for a dataset"
  [fetcher]
  (match [fetcher]
         [(_ :guard seq?)]
         `(.totalOutcomes ~fetcher)
         :else
         (.totalOutcomes fetcher)))
