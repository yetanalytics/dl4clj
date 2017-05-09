(ns dl4clj.eval.evaluation
  (:import [org.deeplearning4j.eval Evaluation RegressionEvaluation])
  (:require [dl4clj.nn.conf.utils :refer [contains-many?]]))


;; regression evaluation
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/RegressionEvaluation.html


;; classification evaluation
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/Evaluation.html


;; TODO

;; add prediction ns
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/meta/Prediction.html

;; add eval tools ns
;; https://deeplearning4j.org/doc/org/deeplearning4j/evaluation/EvaluationTools.html

;; implement binary evaluators
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/EvaluationBinary.html

;; make a seperate ns for the ROC evaluators
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROC.html
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROCMultiClass.html
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/package-summary.html

(defn new-evaluator

  ;; refactor
  "creates an evaluation object for evaling a trained network.

  args:

  :classification? (boolean, default true): determines if a classification or regression
   evaler is created

  :classification? = true args

  -- :n-classes (int): number of classes to account for in the evaluation
  -- :labels (list of strings): the labels to include with the evaluation
  -- :top-n (int): when looking for the top N accuracy of a model
  -- :label-to-idx-map {label-idx (int) label-value (string)}, replaces the use of :labels

  :classification? = false args

  -- :n-columns (int): number of columns in the dataset
  -- :precision (int): specified precision to be returned when you call stats
  -- :column-names (coll of strings): names of the columns"
  [{:keys [classification? n-classes labels
           top-n label-to-idx-map n-columns
           precision column-names]
    :or {classification? true}
    :as opts}]
  (if (true? classification?)
    (cond (and (contains-many? opts :labels :top-n)
               (list? labels)
               (integer? top-n))
          (Evaluation. labels top-n)
          (and (contains? opts :labels)
               (list? labels))
          (Evaluation. labels)
          (and (contains? opts :label-to-idx-map)
               (map? label-to-idx-map))
          (Evaluation. label-to-idx-map)
          (and (contains? opts :n-classes)
               (integer? n-classes))
          (Evaluation. n-classes)
          :else
          (Evaluation.))
    (cond (and (contains-many? opts :column-names :precision)
               (list? column-names)
               (integer? precision))
          (RegressionEvaluation. column-names precision)
          (and (contains-many? opts :n-columns :precision)
               (integer? n-columns)
               (integer? precision))
          (RegressionEvaluation. n-columns precision)
          (and (contains? opts :column-names)
               (or (list? column-names)
                   (> (count column-names) 1)))
          (RegressionEvaluation. column-names)
          (and (contains? opts :n-columns)
               (integer? n-columns))
          (RegressionEvaluation. n-columns)
          :else
          (assert
           false
           "you must supply either the number of columns or their names for regression evaluation"))))

(defn eval-classification
  "depending on args supplied in opts map, does one of:

  - Collects statistics on the real outcomes vs the guesses.
  - Evaluate the output using the given true labels, the input to the multi layer network and the multi layer network to use for evaluation
  - Evaluate the network, with optional metadata
  - Evaluate a single prediction (one prediction at a time)

  1) is accomplished by supplying :real-outcomes and :guesses
  2) is accomplished by supplying :true-labels, :in and :comp-graph or :mln
  3) is accomplished by supplying :real-outcomes, :guesses and :record-meta-data
  4) is accomplished by supplying :predicted-idx and :actual-idx"
  [evaler & {:keys [real-outcomes guesses
                    true-labels in comp-graph
                    record-meta-data mln
                    predicted-idx actual-idx]
             :as opts}]
  (cond (contains-many? opts :true-labels
                        :in :comp-graph)
        (.eval evaler true-labels in comp-graph)
        (contains-many? opts :true-labels
                        :in :mln)
        (.eval evaler true-labels in mln)
        (contains-many? opts :real-outcomes
                        :guesses :record-meta-data)
        (.eval evaler real-outcomes guesses record-meta-data)
        (contains-many? opts :real-outcomes
                        :guesses)
        (.eval evaler real-outcomes guesses)
        (contains-many? opts :predicted-idx :actual-idx)
        (.eval predicted-idx actual-idx)
        :else
        (assert false "you must supply the evaler one of the set of opts described in the doc string"))
  evaler)

(defn eval-time-series
  "evalatues a time series given labels and predictions.

  labels-mask is optional and only applies when there is a mask"
  [evaler & {:keys [labels predicted labels-mask]
             :as opts}]
  (cond (contains? opts :labels-mask)
        (.evalTimeSeries evaler labels predicted labels-mask)
        (false? (contains? opts :labels-mask))
        (.evalTimeSeries evaler labels predicted)
        :else
        (assert false "you must supply labels-mask and/or labels and predicted values"))
  evaler)


(defn get-accuracy
  "Accuracy: (TP + TN) / (P + N)"
  [evaler]
  (.accuracy evaler))

(defn add-to-confusion
  "Adds to the confusion matrix"
  [evaler real-value guess-value]
  (doto evaler
    (.addToConfusion real-value guess-value)))

(defn class-count
  "Returns the number of times the given label has actually occurred"
  [evaler class-label]
  (assert (integer? class-label)
          "class-label needs to be the index (integer) of the label")
  (.classCount evaler class-label))

(defn confusion-to-string
  "Get a String representation of the confusion matrix"
  [evaler]
  (.confusionToString evaler))

(defn f1
  "TP: true positive FP: False Positive FN: False Negative
  F1 score = 2 * TP / (2TP + FP + FN),

  the calculation will only be done for a single class if that classes idx is supplied
   -here class refers to the labels the network was trained on"
  ([evaler]
   (.f1 evaler))
  ([evaler class-label-idx]
   (.f1 evaler class-label-idx)))

(defn false-alarm-rate
  "False Alarm Rate (FAR) reflects rate of misclassified to classified records"
  [evaler]
  (.falseAlarmRate evaler))

(defn false-negative-rate
  "False negative rate based on guesses so far Takes into account all known classes
  and outputs average fnr across all of them

  can be scoped down to a single class if class-label-idx supplied"
  ([evaler]
   (.falseNegativeRate evaler))
  ([evaler class-label-idx]
   (.falseNegativeRate evaler class-label-idx))
  ([evaler class-label-idx edge-case]
   (.falseNegativeRate evaler class-label-idx edge-case)))

(defn false-negatives
  "False negatives: correctly rejected"
  [evaler]
  (.falseNegatives evaler))

(defn false-positive-rate
  "False positive rate based on guesses so far Takes into account all known classes
  and outputs average fpr across all of them

  can be scoped down to a single class if class-label-idx supplied"
  ([evaler]
   (.falsePositiveRate evaler))
  ([evaler class-label-idx]
   (.falsePositiveRate evaler class-label-idx))
  ([evaler class-label-idx edge-case]
   (.falsePositiveRate evaler class-label-idx edge-case)))

(defn false-positives
  "False positive: wrong guess"
  [evaler]
  (.falsePositives evaler))

(defn get-class-label
  "get the class a label is associated with given an idx"
  [evaler label-idx]
  (.getClassLabel evaler label-idx))

(defn get-confusion-matrix
  "Returns the confusion matrix variable"
  [evaler]
  (.getConfusionMatrix evaler))

(defn get-num-row-counter
  [evaler]
  (.getNumRowCounter evaler))

(defn get-prediction-by-predicted-class
  "Get a list of predictions, for all data with the specified predicted class,
  regardless of the actual data class."
  [evaler idx-of-predicted-class]
  (.getPredictionByPredictedClass evaler idx-of-predicted-class))

(defn get-prediction-errors
  "Get a list of prediction errors, on a per-record basis"
  [evaler]
  (.getPredictionErrors evaler))

(defn get-predictions
  "Get a list of predictions in the specified confusion matrix entry
  (i.e., for the given actua/predicted class pair)"
  [evaler actual-class-idx predicted-class-idx]
  (.getPredictions evaler actual-class-idx predicted-class-idx))

(defn get-predictions-by-actual-class
  "Get a list of predictions, for all data with the specified actual class,
  regardless of the predicted class."
  [evaler actual-class-idx]
  (.getPredictionsByActualClass evaler actual-class-idx))

(defn get-top-n-correct-count
  "Return the number of correct predictions according to top N value."
  [evaler]
  (.getTopNCorrectCount evaler))

(defn get-top-n-total-count
  "Return the total number of top N evaluations."
  [evaler]
  (.getTopNTotalCount evaler))

(defn increment-false-negatives
  [evaler class-label-idx]
  (doto evaler
    (.incrementFalseNegatives class-label-idx)))

(defn increment-false-positives
  [evaler class-label-idx]
  (doto evaler
    (.incrementFalsePositives class-label-idx)))

(defn increment-true-negatives
  [evaler class-label-idx]
  (doto evaler
    (.incrementTrueNegatives class-label-idx)))

(defn increment-true-positives
  [evaler class-label-idx]
  (doto evaler
    (.incrementTruePositives class-label-idx)))

(defn merge-evals
  "Merge the other evaluation object into evaler"
  [evaler other-evaler]
  (doto evaler
    (.merge other-evaler)))

(defn total-negatives
  "Total negatives true negatives + false negatives"
  [evaler]
  (.negative evaler))

(defn total-positives
  "Returns all of the positive guesses: true positive + false negative"
  [evaler]
  (.positive evaler))

(defn get-precision
  "Precision based on guesses so far Takes into account all known classes and
  outputs average precision across all of them.

  can be scoped to a label given its idx"
  ([evaler]
   (.precision evaler))
  ([evaler class-label-idx]
   (.precision evaler class-label-idx))
  ([evaler class-label-idx edge-case]
   (.precision evaler class-label-idx edge-case)))

(defn recall
  "Recall based on guesses so far Takes into account all known classes
  and outputs average recall across all of them

  can be scoped to a label given its idx"
  ([evaler]
   (.recall evaler))
  ([evaler class-label-idx]
   (.recall evaler class-label-idx))
  ([evaler class-label-idx edge-case]
   (.recall evaler class-label-idx edge-case)))

(defn get-stats
  "Method to obtain the classification report as a String"
  ([evaler]
   (.stats evaler))
  ([evaler suppress-warnings?]
   (.stats evaler suppress-warnings?)))

(defn top-n-accuracy
  "Top N accuracy of the predictions so far."
  [evaler]
  (.topNAccuracy evaler))

(defn true-negatives
  "True negatives: correctly rejected"
  [evaler]
  (.trueNegatives evaler))

(defn true-positives
  "True positives: correctly rejected"
  [evaler]
  (.truePositives evaler))

(defn get-mean-squared-error
  "returns the MSE"
  [regression-evaler column-idx]
  (.meanSquaredError regression-evaler column-idx))

(defn get-mean-absolute-error
  "returns MAE"
  [regression-evaler column-idx]
  (.meanAbsoluteError regression-evaler column-idx))

(defn get-root-mean-squared-error
  "returns rMSE"
  [regression-evaler column-idx]
  (.rootMeanSquaredError regression-evaler column-idx))

(defn get-correlation-r2
  "return the R2 correlation"
  [regression-evaler column-idx]
  (.correlationR2 regression-evaler column-idx))

(defn get-relative-squared-error
  "return relative squared error"
  [regression-evaler column-idx]
  (.relativeSquaredError regression-evaler column-idx))
