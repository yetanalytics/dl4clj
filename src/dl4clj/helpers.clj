(ns dl4clj.helpers
  (:require [nd4clj.linalg.api.ds-iter :refer [get-current-cursor
                                               reset-iter!
                                               has-next?
                                               next-example!]]))

(defn reset-if-not-at-start!
  "checks the current cursor of the iterator and if not at 0 resets it"
  [iter]
  (if (not= 0 (get-current-cursor iter))
    (reset-iter! iter)
    iter))

(defn reset-if-empty?!
  "resets an iterator if we are at the end"
  [iter]
  (if (false? (has-next? iter))
    (reset-iter! iter)
    iter))

(defn data-from-iter
  "returns all the data from an iterator as a lazy seq"
  [iter]
  (when (has-next? iter)
    (lazy-seq (cons (next-example! iter) (data-from-iter iter)))))

(defn new-lazy-iter
  "creates a barebones iterator for a lazy seq"
  [lazy-data]
  (.iterator lazy-data))
