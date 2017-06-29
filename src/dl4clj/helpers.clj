(ns dl4clj.helpers
  (:require [dl4clj.datasets.api.iterators :refer [get-current-cursor
                                                   reset-iter!
                                                   has-next?
                                                   next-example!]]))

(defn reset-if-empty?!
  "resets an iterator if we are at the end"
  [iter]
  (if (false? (has-next? iter))
    (reset-iter! iter)
    iter))

(defn reset-iterator!
  "resets an iterator, won't reset a lazy iter but will return it"
  [iter]
  (try (reset-iter! iter)
       (catch Exception e iter)))

(defn data-from-iter
  "returns all the data from an iterator as a lazy seq"
  [iter]
  (when (has-next? iter)
    (lazy-seq (cons (next-example! iter) (data-from-iter iter)))))

(defn new-lazy-iter
  "creates a barebones iterator for a lazy seq"
  [lazy-data]
  (.iterator lazy-data))
