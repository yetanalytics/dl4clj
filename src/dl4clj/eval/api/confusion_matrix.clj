(ns ^{:doc "implementation of the ConfusionMatrix class api methods in dl4j.
 see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/ConfusionMatrix.html"}
    dl4clj.eval.api.confusion-matrix
  (:import [org.deeplearning4j.eval ConfusionMatrix])
  (:require [clojure.core.match :refer [match]]
            [dl4clj.utils :refer [obj-or-code? eval-if-code]]))

(defn get-actual-total
  "Computes the total number of times the class actually appeared in the data."
  [& {:keys [confusion-matrix actual as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:confusion-matrix (_ :guard seq?)
           :actual (_ :guard seq?)}]
         (obj-or-code? as-code? `(.getActualTotal ~confusion-matrix ~actual))
         :else
         (let [[m-obj a-obj] (eval-if-code [confusion-matrix seq?]
                                           [actual seq?])]
           (.getActualTotal m-obj a-obj))))

(defn get-classes
  "Gives the applyTransformToDestination of all classes in the confusion matrix."
  [confusion-matrix & {:keys [as-code?]
                       :or {as-code? true}}]
  (match [confusion-matrix]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getClasses ~confusion-matrix))
         :else
         (.getClasses confusion-matrix)))

(defn get-count
  "Gives the count of the number of times the predicted class was predicted for the actual class."
  [& {:keys [confusion-matrix actual predicted as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:confusion-matrix (_ :guard seq?)
           :actual (_ :guard seq?)
           :predicted (_ :guard seq?)}]
         (obj-or-code? as-code? `(.getCount ~confusion-matrix ~actual ~predicted))
         :else
         (let [[m-obj a-obj p-obj] (eval-if-code [confusion-matrix seq?]
                                                 [actual seq?]
                                                 [predicted seq?])]
           (.getCount m-obj a-obj p-obj))))

(defn get-predicted-total
  "Computes the total number of times the class was predicted by the classifier."
  [& {:keys [confusion-matrix predicted as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:confusion-matrix (_ :guard seq?)
           :predicted (_ :guard seq?)}]
         (obj-or-code? as-code? `(.getPredictedTotal ~confusion-matrix ~predicted))
         :else
         (let [[m-obj p-obj] (eval-if-code [confusion-matrix seq?]
                                           [predicted seq?])]
           (.getPredictedTotal m-obj p-obj))))

(defn to-csv
  "Outputs the ConfusionMatrix as comma-separated values for easy import into spreadsheets"
  [confusion-matrix & {:keys [as-code?]
                       :or {as-code? true}}]
  (match [confusion-matrix]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.toCSV ~confusion-matrix))
         :else
         (.toCSV confusion-matrix)))

(defn to-html
  "Outputs Confusion Matrix in an HTML table."
  [confusion-matrix & {:keys [as-code?]
                       :or {as-code? true}}]
  (match [confusion-matrix]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.toHTML ~confusion-matrix))
         :else
         (.toHTML confusion-matrix)))
