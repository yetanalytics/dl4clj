(ns ^{:doc "The VisibleUnit, or layer, is the layer of nodes where input goes in,
            and the HiddenUnit is the layer where those inputs are recombined in
            ore complex features.  see https://deeplearning4j.org/restrictedboltzmannmachine"}
    dl4clj.nn.conf.builders.rbm-units
  (:require [clojure.string :as s])
  (:import [org.deeplearning4j.nn.conf.layers RBM$VisibleUnit RBM$HiddenUnit]))

(defn hidden-value-of [k]
  (if (string? k)
    (RBM$HiddenUnit/valueOf k)
    (RBM$HiddenUnit/valueOf (s/replace (s/upper-case (name k)) "-" "_"))))

(defn visible-value-of [k]
  (if (string? k)
    (RBM$VisibleUnit/valueOf k)
    (RBM$VisibleUnit/valueOf (s/replace (s/upper-case (name k)) "-" "_"))))

(defn hidden-values []
  (map #(keyword (s/replace (s/lower-case (str %)) "_" "-")) (RBM$HiddenUnit/values)))

(defn visible-values []
  (map #(keyword (s/replace (s/lower-case (str %)) "_" "-")) (RBM$VisibleUnit/values)))

(comment
  (map hidden-value-of (hidden-values))
  (hidden-value-of :softmax)
  (visible-value-of :softmax)

  (hidden-value-of :binary)
  (visible-value-of :binary)

  (hidden-value-of :gaussian)
  (visible-value-of :gaussian)

  (hidden-value-of :identity)
  (visible-value-of :identity)

  (hidden-value-of :rectified)
  (visible-value-of :linear)
)
