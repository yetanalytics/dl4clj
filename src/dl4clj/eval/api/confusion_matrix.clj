(ns ^{:doc "implementation of the ConfusionMatrix class api methods in dl4j.
 see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/ConfusionMatrix.html"}
    dl4clj.eval.api.confusion-matrix
  (:import [org.deeplearning4j.eval ConfusionMatrix])
  (:require [clojure.core.match :refer [match]]))

(defn get-actual-total
  "Computes the total number of times the class actually appeared in the data."
  [& {:keys [confusion-matrix actual]
      :as opts}]
  (match [opts]
         [{:confusion-matrix (_ :guard seq?)
           :actual (_ :guard seq?)}]
         `(.getActualTotal ~confusion-matrix ~actual)
         :else
         (.getActualTotal confusion-matrix actual)))

(defn get-classes
  "Gives the applyTransformToDestination of all classes in the confusion matrix."
  [confusion-matrix]
  (match [confusion-matrix]
         [(_ :guard seq?)]
         `(.getClasses ~confusion-matrix)
         :else
         (.getClasses confusion-matrix)))

(defn get-count
  "Gives the count of the number of times the predicted class was predicted for the actual class."
  [& {:keys [confusion-matrix actual predicted]
      :as opts}]
  (match [opts]
         [{:confusion-matrix (_ :guard seq?)
           :actual (_ :guard seq?)
           :predicted (_ :guard seq?)}]
         `(.getCount ~confusion-matrix ~actual ~predicted)
         :else
         (.getCount confusion-matrix actual predicted)))

(defn get-predicted-total
  "Computes the total number of times the class was predicted by the classifier."
  [& {:keys [confusion-matrix predicted]
      :as opts}]
  (match [opts]
         [{:confusion-matrix (_ :guard seq?)
           :predicted (_ :guard seq?)}]
         `(.getPredictedTotal ~confusion-matrix ~predicted)
         :else
         (.getPredictedTotal confusion-matrix predicted)))

(defn to-csv
  "Outputs the ConfusionMatrix as comma-separated values for easy import into spreadsheets"
  [confusion-matrix]
  (match [confusion-matrix]
         [(_ :guard seq?)]
         `(.toCSV ~confusion-matrix)
         :else
         (.toCSV confusion-matrix)))

(defn to-html
  "Outputs Confusion Matrix in an HTML table."
  [confusion-matrix]
  (match [confusion-matrix]
         [(_ :guard seq?)]
         `(.toHTML ~confusion-matrix)
         :else
         (.toHTML confusion-matrix)))
