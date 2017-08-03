(ns dl4clj.helpers
  (:require [dl4clj.datasets.api.iterators :refer [get-current-cursor
                                                   reset-iter!
                                                   has-next?
                                                   next-example!]]
            [dl4clj.constants :as constants]
            [dl4clj.nn.conf.distributions :as distribution]
            [dl4clj.nn.conf.step-fns :as step-functions]
            [dl4clj.nn.conf.input-pre-processor :as pre-process]))

;; update helpers to accept fn calls (the ones that can)
;; not just config maps
;; should just be somthing like
#_(match [input]
         [(_ :guard seq?)] `(eval passed-in-fn)
         :else
         do whatever was already being done)
;; this may have to be within a forloop in some cases
;; like when the input is a collection of to-be java objects

(defn pre-processor-helper
  [pps]
  (into {}
        (for [each pps
              :let [[idx pp] each]]
          {idx `(pre-process/pre-processors ~pp)})))

(defn value-of-helper
  [k v]
  `(constants/value-of {~k ~v}))

(defn distribution-helper
  [opts]
  `(distribution/distribution ~opts))

(defn step-fn-helper
  [opts]
  `(step-functions/step-fn ~opts))

(defn input-type-helper
  [input-type]
  `(constants/input-types ~input-type))

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
