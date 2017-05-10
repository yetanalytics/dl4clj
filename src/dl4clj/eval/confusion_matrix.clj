(ns ^{:doc "implementation of the ConfusionMatrix class in dl4j.
 see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/ConfusionMatrix.html"}
    dl4clj.eval.confusion-matrix
  (:import [org.deeplearning4j.eval ConfusionMatrix])
  (:require [dl4clj.utils :refer [contains-many?]]))

;;https://deeplearning4j.org/doc/org/deeplearning4j/eval/ConfusionMatrix.html

(defn new-confusion-matrix
  [& {:keys [existing-confusion-matrix classes]
      :as opts}]
  (cond (contains? opts :classes)
        (ConfusionMatrix. classes)
        (contains? opts :existing-confusion-matrix)
        (ConfusionMatrix. existing-confusion-matrix)
        :else
        (assert false "you must provide a list of classes or an existing confusion matrix to create a new one")))

(defn add!
  [& {:keys [base-confusion-matrix other-confusion-matrix actual predicted n]
      :as opts}]
  (cond (contains-many? opts :base-confusion-matrix :actual :predicted :n)
        (doto base-confusion-matrix (.add actual predicted n))
        (contains-many? opts :base-confusion-matrix :actual :predicted)
        (doto base-confusion-matrix (.add actual predicted))
        (contains-many? opts :other-confusion-matrix :base-confusion-matrix)
        (doto base-confusion-matrix (.add other-confusion-matrix))
        :else
        (assert false "you must supply the data to add to the base confusion matrix")))

(defn get-actual-total
  "Computes the total number of times the class actually appeared in the data."
  [& {:keys [confusion-matrix actual]}]
  (.getActualTotal confusion-matrix actual))

(defn get-classes
  "Gives the applyTransformToDestination of all classes in the confusion matrix."
  [confusion-matrix]
  (.getClasses confusion-matrix))

(defn get-count
  "Gives the count of the number of times the predicted class was predicted for the "actual" class."
  [& {:keys [confusion-matrix actual predicted]}]
  (.getCount confusion-matrix actual predicted))

(defn get-predicted-total
  "Computes the total number of times the class was predicted by the classifier."
  [& {:keys [confusion-matrix predicted]}]
  (.getPredictedTotal confusion-matrix predicted))

(defn to-csv
  "Outputs the ConfusionMatrix as comma-separated values for easy import into spreadsheets"
  [confusion-matrix]
  (.toCSV confusion-matrix))

(defn to-html
  "Outputs Confusion Matrix in an HTML table."
  [confusion-matrix]
  (.toHTML confusion-matrix))
