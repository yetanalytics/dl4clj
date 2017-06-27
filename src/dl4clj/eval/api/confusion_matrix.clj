(ns ^{:doc "implementation of the ConfusionMatrix class api methods in dl4j.
 see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/ConfusionMatrix.html"}
    dl4clj.eval.api.confusion-matrix
  (:import [org.deeplearning4j.eval ConfusionMatrix])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn add!
  "adds things to a confusion matrix

  :other-confusion-matrix (obj), a confusion matrix whose entries will be added to
   the base confusion matrix

  :actual (comparable), specifies an entry in the confusion matrix

  :predicted (compareable), specifies an entry in the confusion matrix

  :n (int), the amount to increment actual and predicted by, defaults to 1"
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
  "Gives the count of the number of times the predicted class was predicted for the actual class."
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
