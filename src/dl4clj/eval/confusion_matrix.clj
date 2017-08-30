(ns ^{:doc "implementation of the ConfusionMatrix class in dl4j.
 see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/ConfusionMatrix.html"}
    dl4clj.eval.confusion-matrix
  (:import [org.deeplearning4j.eval ConfusionMatrix])
  (:require [dl4clj.utils :refer [contains-many?]]))

;; going to be removed in core
(defn new-confusion-matrix
  "Creates a new confusion matrix.

  :existing-confusion-matrix (obj), an existing confusion matrix

  :classes (coll), a collection of java classes which extend java.lang.Comparable"
  [& {:keys [existing-confusion-matrix classes]
      :as opts}]
  (cond (contains? opts :classes)
        (ConfusionMatrix. classes)
        (contains? opts :existing-confusion-matrix)
        (ConfusionMatrix. existing-confusion-matrix)
        :else
        (assert false "you must provide a list of classes or an existing confusion matrix to create a new one")))
