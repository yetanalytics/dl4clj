(ns dl4clj.datasets.api.fetchers
  (:import [org.deeplearning4j.datasets.fetchers BaseDataFetcher])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code?]]))

(defn fetcher-cursor
  "Direct access to a number represenative of iterating through a dataset"
  [& {:keys [fetcher as-code?]
      :or {as-code? true}}]
  (match [fetcher]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.cursor ~fetcher))
         :else
         (.cursor fetcher)))

(defn has-more?
  "returns true if there are more receords left to go through in a dataset"
  [& {:keys [fetcher as-code?]
      :or {as-code? true}}]
  (match [fetcher]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.hasMore ~fetcher))
         :else
         (.hasMore fetcher)))

(defn input-column-length
  "The length of a feature vector for an individual example"
  [& {:keys [fetcher as-code?]
      :or {as-code? true}}]
  (match [fetcher]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.inputColumns ~fetcher))
         :else
         (.inputColumns fetcher)))

(defn reset-fetcher!
  "Returns the fetcher back to the beginning of the dataset, returns the fetcher"
  [& {:keys [fetcher as-code?]
      :or {as-code? true}}]
  (match [fetcher]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~fetcher .reset))
         :else
         (doto fetcher .reset)))

(defn n-examples-in-ds
  "The total number of examples"
  [& {:keys [fetcher as-code?]
      :or {as-code? true}}]
  (match [fetcher]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.totalExamples ~fetcher))
         :else
         (.totalExamples fetcher)))

(defn n-outcomes-in-ds
  "The number of labels for a dataset"
  [& {:keys [fetcher as-code?]
      :or {as-code? true}}]
  (match [fetcher]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.totalOutcomes ~fetcher))
         :else
         (.totalOutcomes fetcher)))
