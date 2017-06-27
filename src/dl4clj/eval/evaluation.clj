(ns ^{:doc "implementation of the eval class in dl4j.  Used to get performance metrics for a model
see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/Evaluation.html and
https://deeplearning4j.org/doc/org/deeplearning4j/eval/RegressionEvaluation.html"}
    dl4clj.eval.evaluation
  (:import [org.deeplearning4j.eval Evaluation RegressionEvaluation BaseEvaluation])
  (:require [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multimethod for creating the evaluation java object
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti evaler generic-dispatching-fn)

(defmethod evaler :classification [opts]
  (let [conf (:classification opts)
        {labels :labels
         top-n :top-n
         l-to-i-map :label-to-idx
         n-classes :n-classes} conf]
    (cond (contains-many? conf :labels :top-n)
          (Evaluation. labels top-n)
          (contains? conf :labels)
          (Evaluation. (into '() labels))
          (contains? conf :label-to-idx)
          (Evaluation. l-to-i-map)
          (contains? conf :n-classes)
          (Evaluation. n-classes)
          :else
          (Evaluation.))))

(defmethod evaler :regression [opts]
  (let [conf (:regression opts)
        {column-names :column-names
         precision :precision
         n-columns :n-columns} conf
        c-names (into '() column-names)]
    (cond (contains-many? conf :column-names :precision)
          (RegressionEvaluation. c-names precision)
          (contains-many? conf :n-columns :precision)
          (RegressionEvaluation. n-columns precision)
          (contains? conf :column-names)
          (RegressionEvaluation. c-names)
          (contains? conf :n-columns)
          (RegressionEvaluation. n-columns)
          :else
          (assert
           false
           "you must supply either the number of columns or their names for regression evaluation"))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-classification-evaler
  "Creates an instance of an evaluation object which reports precision, recall, f1

   :labels (coll), a collection of string labels to use for the output

   :top-n (int), value to use for the top N accuracy calc.
     - An example is considered correct if the probability for the true class
       is one of the highest n values

   :n-classes (int), the number of classes to account for in the evaluation

   :label-to-idx (map), {column-idx (int) label (str)}
    - another way to set the labels for the classification"
  [& {:keys [labels top-n label-to-idx n-classes]
      :as opts}]
  (evaler {:classification opts}))

(defn new-regression-evaler
  "Evaluation method for the evaluation of regression algorithms.

   provides MSE, MAE, RMSE, RSE, correlation coefficient for each column

   :column-names (coll), a collection of string naming the columns

   :precision (int), specified precision to be used

   :n-columns (int), the number of columns in the dataset"
  [& {:keys [column-names precision n-columns]
      :as opts}]
  (evaler {:regression opts}))
