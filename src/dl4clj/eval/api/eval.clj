(ns dl4clj.eval.api.eval
  (:import [org.deeplearning4j.eval Evaluation RegressionEvaluation BaseEvaluation
            IEvaluation])
  (:require [dl4clj.utils :refer [contains-many? get-labels]]
            [dl4clj.datasets.api.iterators :refer [has-next? next-example!]]
            ;; this is going to change
            [dl4clj.nn.api.multi-layer-network :refer [output]]
            [dl4clj.datasets.api.datasets :refer [get-features]]
            [dl4clj.helpers :refer [reset-iterator! new-lazy-iter]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; interact with a classification evaluator
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn eval-classification!
  "evaluate the output of a network.

  :labels (vec or INDArray), the actual labels of the data (target labels)

  :network-predictions (vec or INDArray), the output of the network

  :mask-array (vec or INDArray), the mask array for the data if there is one

  :record-meta-data (coll) meta data that extends java.io.Serializable

  NOTE: for evaluating classification problems, use eval-classification! in
   dl4clj.eval.evaluation, (when the evaler is created by new-classification-evaler)"
  ;; update this docstring
  [& {:keys [labels network-predictions mask-array record-meta-data evaler
             mln features predicted-idx actual-idx]
      :as opts}]
  (assert (contains? opts :evaler) "you must provide an evaler to evaluate a classification task")
  (let [l (vec-or-matrix->indarray labels)
        np (vec-or-matrix->indarray network-predictions)]
   (cond (contains-many? opts :labels :network-predictions :record-meta-data)
        (doto evaler (.eval l np (into '() record-meta-data)))
        (contains-many? opts :labels :network-predictions :mask-array)
        (doto evaler (.eval l np (vec-or-matrix->indarray mask-array)))
        (contains-many? opts :labels :features :mln)
        (doto evaler (.eval l (vec-or-matrix->indarray features) mln))
        (contains-many? opts :labels :network-predictions)
        (doto evaler (.eval l np))
        (contains-many? opts :predicted-idx :actual-idx)
        (doto evaler (.eval predicted-idx actual-idx))
        :else
        (assert false "you must supply an evaler, the correct labels and the network predicted labels"))))

(defn eval-time-series!
  "evalatues a time series given labels and predictions.

  labels-mask is optional and only applies when there is a mask"
  [& {:keys [labels predicted labels-mask evaler]
      :as opts}]
  (let [l (vec-or-matrix->indarray labels)
        p (vec-or-matrix->indarray predicted)]
    (cond (contains? opts :labels-mask)
          (doto evaler (.evalTimeSeries l p (vec-or-matrix->indarray labels-mask)))
          (false? (contains? opts :labels-mask))
          (doto evaler (.evalTimeSeries l p))
          :else
          (assert false "you must supply labels-mask and/or labels and predicted values"))))

(defn get-stats
  "Method to obtain the classification report as a String"
  [& {:keys [evaler suppress-warnings?]
      :as opts}]
  (if (contains? opts :suppress-warnings?)
    (.stats evaler suppress-warnings?)
    (.stats evaler)))

(defn eval-model-whole-ds
  "evaluate the model performance on an entire data set and print the final result

  :mln (multi layer network), a trained mln you want to get classification stats for

  :eval-obj (evaler), the object created by new-classification-evaler

  :iter (iter), the dataset iterator which has the data you want to evaluate the model on

  :lazy-data (lazy-seq), a lazy sequence of dataset objects

  you should supply either a dl4j dataset-iterator (:iter) or a lazy-seq (:lazy-data), not both

  returns the evaluation object"
  [& {:keys [mln evaler iter lazy-data]
      :as opts}]
  (let [ds-iter (if (contains? opts :lazy-data)
                  (new-lazy-iter lazy-data)
                  (reset-iterator! iter))]
    (while (has-next? ds-iter)
      (let [nxt (next-example! ds-iter)
            prediction (output :mln mln :input (get-features nxt))]
        (eval-classification!
         :evaler evaler
         :labels (get-labels nxt)
         :network-predictions prediction))))
  (println (get-stats :evaler evaler))
  evaler)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; classification evaluator interaction fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-accuracy
  "Accuracy: (TP + TN) / (P + N)"
  [evaler]
  (.accuracy evaler))

(defn add-to-confusion
  ;; shouldn't be a user facing fn
  ;; will be removed in the core branch
  "Adds to the confusion matrix"
  [& {:keys [evaler real-value guess-value]}]
  (doto evaler
    (.addToConfusion (int real-value) (int guess-value))))

(defn class-count
  "Returns the number of times the given label has actually occurred"
  [& {:keys [evaler class-label-idx]}]
  (.classCount evaler (int class-label-idx)))

(defn confusion-to-string
  "Get a String representation of the confusion matrix"
  [evaler]
  (.confusionToString evaler))

(defn f1
  "TP: true positive FP: False Positive FN: False Negative
  F1 score = 2 * TP / (2TP + FP + FN),

  the calculation will only be done for a single class if that classes idx is supplied
   -here class refers to the labels the network was trained on"
  [& {:keys [evaler class-label-idx]
      :as opts}]
  (if (contains? opts :class-label-idx)
    (.f1 evaler (int class-label-idx))
    (.f1 evaler)))

(defn false-alarm-rate
  "False Alarm Rate (FAR) reflects rate of misclassified to classified records"
  [evaler]
  (.falseAlarmRate evaler))

(defn false-negative-rate
  "False negative rate based on guesses so far Takes into account all known classes
  and outputs average fnr across all of them

  can be scoped down to a single class if class-label-idx supplied"
  [& {:keys [evaler class-label-idx edge-case]
      :as opts}]
  (cond (contains-many? opts :class-label-idx :edge-case :evaler)
        (.falseNegativeRate evaler (int class-label-idx) edge-case)
        (contains-many? opts :evaler :class-label-idx)
        (.falseNegativeRate evaler (int class-label-idx))
        (contains? opts :evaler)
        (.falseNegativeRate evaler)
        :else
        (assert false "you must atleast provide an evaler to get the false negative rate of the model being evaluated")))

(defn false-negatives
  "False negatives: correctly rejected"
  [evaler]
  (.falseNegatives evaler))

(defn false-positive-rate
  "False positive rate based on guesses so far Takes into account all known classes
  and outputs average fpr across all of them

  can be scoped down to a single class if class-label-idx supplied"
  [& {:keys [evaler class-label-idx edge-case]
      :as opts}]
  (cond (contains-many? opts :class-label-idx :edge-case :evaler)
        (.falsePositiveRate evaler (int class-label-idx) edge-case)
        (contains-many? opts :evaler :class-label-idx)
        (.falsePositiveRate evaler (int class-label-idx))
        (contains? opts :evaler)
        (.falsePositiveRate evaler)
        :else
        (assert false "you must atleast provide an evaler to get the false positive rate of the model being evaluated")))

(defn false-positives
  "False positive: wrong guess"
  [evaler]
  (.falsePositives evaler))

(defn get-class-label
  "get the class a label is associated with given an idx"
  [& {:keys [evaler label-idx]}]
  (.getClassLabel evaler (int label-idx)))

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
  [& {:keys [evaler idx-of-predicted-class]}]
  (.getPredictionByPredictedClass evaler (int idx-of-predicted-class)))

(defn get-prediction-errors
  "Get a list of prediction errors, on a per-record basis"
  [evaler]
  (.getPredictionErrors evaler))

(defn get-predictions
  "Get a list of predictions in the specified confusion matrix entry
  (i.e., for the given actua/predicted class pair)"
  [& {:keys [evaler actual-class-idx predicted-class-idx]}]
  (.getPredictions evaler actual-class-idx predicted-class-idx))

(defn get-predictions-by-actual-class
  "Get a list of predictions, for all data with the specified actual class,
  regardless of the predicted class."
  [& {:keys [evaler actual-class-idx]}]
  (.getPredictionsByActualClass evaler actual-class-idx))

(defn get-top-n-correct-count
  "Return the number of correct predictions according to top N value."
  [evaler]
  (.getTopNCorrectCount evaler))

(defn get-top-n-total-count
  "Return the total number of top N evaluations."
  [evaler]
  (.getTopNTotalCount evaler))

(defn increment-false-negatives!
  ;; should not be a user facing fn
  ;; will be removed in the core branch
  [& {:keys [evaler class-label-idx]}]
  (doto evaler
    (.incrementFalseNegatives (int class-label-idx))))

(defn increment-false-positives!
  ;; should not be a user facing fn
  ;; will be removed in the core branch
  [& {:keys [evaler class-label-idx]}]
  (doto evaler
    (.incrementFalsePositives (int class-label-idx))))

(defn increment-true-negatives!
  ;; should not be a user facing fn
  ;; will be removed in the core branch
  [& {:keys [evaler class-label-idx]}]
  (doto evaler
    (.incrementTrueNegatives (int class-label-idx))))

(defn increment-true-positives!
  ;; should not be a user facing fn
  ;; will be removed in the core branch
  [& {:keys [evaler class-label-idx]}]
  (doto evaler
    (.incrementTruePositives (int class-label-idx))))

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
  [& {:keys [evaler class-label-idx edge-case]
      :as opts}]
  (cond (contains-many? opts :class-label-idx :edge-case :evaler)
        (.precision evaler (int class-label-idx) edge-case)
        (contains-many? opts :evaler :class-label-idx)
        (.precision evaler (int class-label-idx))
        (contains? opts :evaler)
        (.precision evaler)
        :else
        (assert false "you must atleast provide an evaler to get the precision of the model being evaluated")))

(defn recall
  "Recall based on guesses so far Takes into account all known classes
  and outputs average recall across all of them

  can be scoped to a label given its idx"
  [& {:keys [evaler class-label-idx edge-case]
      :as opts}]
  (cond (contains-many? opts :class-label-idx :edge-case :evaler)
        (.recall evaler (int class-label-idx) edge-case)
        (contains-many? opts :evaler :class-label-idx)
        (.recall evaler (int class-label-idx))
        (contains? opts :evaler)
        (.recall evaler)
        :else
        (assert false "you must atleast provide an evaler to get the recall of the model being evaluated")))

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; interact with a regression evaluator
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-mean-squared-error
  "returns the MSE"
  [& {:keys [regression-evaler column-idx]}]
  (.meanSquaredError regression-evaler column-idx))

(defn get-mean-absolute-error
  "returns MAE"
  [& {:keys [regression-evaler column-idx]}]
  (.meanAbsoluteError regression-evaler column-idx))

(defn get-root-mean-squared-error
  "returns rMSE"
  [& {:keys [regression-evaler column-idx]}]
  (.rootMeanSquaredError regression-evaler column-idx))

(defn get-correlation-r2
  "return the R2 correlation"
  [& {:keys [regression-evaler column-idx]}]
  (.correlationR2 regression-evaler column-idx))

(defn get-relative-squared-error
  "return relative squared error"
  [& {:keys [regression-evaler column-idx]}]
  (.relativeSquaredError regression-evaler column-idx))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; general
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn merge!
  "merges objects that implemented the IEvaluation interface

  evaler and other-evaler can be evaluations, ROCs or MultiClassRocs"
  [& {:keys [evaler other-evaler]}]
  (doto evaler (.merge other-evaler)))
