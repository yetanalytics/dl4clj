(ns dl4clj.eval.api.eval
  (:import [org.deeplearning4j.eval Evaluation RegressionEvaluation BaseEvaluation
            IEvaluation])
  (:require [dl4clj.utils :refer [contains-many? get-labels]]
            [dl4clj.datasets.api.iterators :refer [has-next? next-example!]]
            [dl4clj.nn.api.multi-layer-network :refer [output]]
            [dl4clj.datasets.api.datasets :refer [get-features]]
            [dl4clj.helpers :refer [reset-iterator! new-lazy-iter]]
            [clojure.core.match :refer [match]]
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
  (match [opts]
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :network-predictions (:or (_ :guard vector?)
                                     (_ :guard seq?))
           :record-meta-data (_ :guard seq?)}]
         `(doto ~evaler
            (.eval (vec-or-matrix->indarray ~labels)
                   (vec-or-matrix->indarray ~network-predictions)
                   (into '() ~record-meta-data)))
         [{:evaler _
           :labels _
           :network-predictions _
           :record-meta-data _}]
         (doto evaler
            (.eval (vec-or-matrix->indarray labels)
                   (vec-or-matrix->indarray network-predictions)
                   (into '() record-meta-data)))
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :network-predictions (:or (_ :guard vector?)
                                     (_ :guard seq?))
           :mask-array (:or (_ :guard vector?)
                            (_ :guard seq?))}]
         `(doto ~evaler
           (.eval (vec-or-matrix->indarray ~labels)
                  (vec-or-matrix->indarray ~network-predictions)
                  (vec-or-matrix->indarray ~mask-array)))
         [{:evaler _
           :labels _
           :network-predictions _
           :mask-array _}]
         (doto evaler
            (.eval (vec-or-matrix->indarray labels)
                   (vec-or-matrix->indarray network-predictions)
                   (vec-or-matrix->indarray mask-array)))
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))
           :mln (_ :guard seq?)}]
         `(doto ~evaler (.eval (vec-or-matrix->indarray ~labels)
                               (vec-or-matrix->indarray ~features)
                               ~mln))
         [{:evaler _
           :labels _
           :features _
           :mln _}]
         (doto evaler (.eval (vec-or-matrix->indarray labels)
                             (vec-or-matrix->indarray features)
                             mln))
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :network-predictions (:or (_ :guard vector?)
                                     (_ :guard seq?))}]
         `(doto ~evaler
            (.eval (vec-or-matrix->indarray ~labels)
                   (vec-or-matrix->indarray ~network-predictions)))
         [{:evaler _
           :labels _
           :network-predictions _}]
         (doto evaler
            (.eval (vec-or-matrix->indarray labels)
                   (vec-or-matrix->indarray network-predictions)))
         [{:evaler (_ :guard seq?)
           :predicted-idx (:or (_ :guard number?)
                               (_ :guard seq?))
           :actual-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         `(doto ~evaler (.eval (int ~predicted-idx) (int ~actual-idx)))
         [{:evaler _
           :predicted-idx _
           :actual-idx _}]
         (doto evaler (.eval predicted-idx actual-idx))))

(defn eval-time-series!
  "evalatues a time series given labels and predictions.

  labels-mask is optional and only applies when there is a mask"
  [& {:keys [labels predicted labels-mask evaler]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :predicted (:or (_ :guard vector?)
                           (_ :guard seq?))
           :labels-mask (:or (_ :guard vector?)
                             (_ :guard seq?))}]
         `(doto ~evaler (.evalTimeSeries
                         (vec-or-matrix->indarray ~labels)
                         (vec-or-matrix->indarray ~predicted)
                         (vec-or-matrix->indarray ~labels-mask)))
         [{:evaler _
           :labels _
           :predicted _
           :labels-mask _}]
         (doto evaler (.evalTimeSeries
                         (vec-or-matrix->indarray labels)
                         (vec-or-matrix->indarray predicted)
                         (vec-or-matrix->indarray labels-mask)))
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :predicted (:or (_ :guard vector?)
                           (_ :guard seq?))}]
         `(doto ~evaler (.evalTimeSeries
                         (vec-or-matrix->indarray ~labels)
                         (vec-or-matrix->indarray ~predicted)))
         [{:evaler _
           :labels _
           :predicted _}]
         (doto evaler (.evalTimeSeries
                       (vec-or-matrix->indarray labels)
                       (vec-or-matrix->indarray predicted)))))

(defn get-stats
  "Method to obtain the classification report as a String"
  [& {:keys [evaler suppress-warnings?]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :suppress-warnings? (:or (_ :guard boolean?)
                                    (_ :guard seq?))}]
         `(.stats ~evaler ~suppress-warnings?)
         [{:evaler _
           :suppress-warnings? _}]
         (.stats evaler suppress-warnings?)
         [{:evaler (_ :guard seq?)}]
         (.stats evaler)
         [{:evaler _}]
         (.stats evaler)))

(defn eval-model-whole-ds
  ;; move to core, update then
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
  (match [evaler]
         [(_ :guard seq?)]
         `(.accuracy ~evaler)
         :else
         (.accuracy evaler)))

(defn class-count
  "Returns the number of times the given label has actually occurred"
  [& {:keys [evaler class-label-idx]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         `(.classCount ~evaler (int ~class-label-idx))
         :else
         (.classCount evaler (int class-label-idx))))

(defn confusion-to-string
  "Get a String representation of the confusion matrix"
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.confusionToString ~evaler)
         :else
         (.confusionToString evaler)))

(defn f1
  "TP: true positive FP: False Positive FN: False Negative
  F1 score = 2 * TP / (2TP + FP + FN),

  the calculation will only be done for a single class if that classes idx is supplied
   -here class refers to the labels the network was trained on"
  [& {:keys [evaler class-label-idx]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         `(.f1 ~evaler (int ~class-label-idx))
         [{:evaler _
           :class-label-idx _ }]
         (.f1 evaler (int class-label-idx))
         [{:evaler (_ :guard seq?)}]
         `(.f1 ~evaler)
         [{:evaler _}]
         (.f1 evaler)))

(defn false-alarm-rate
  "False Alarm Rate (FAR) reflects rate of misclassified to classified records"
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.falseAlarmRate ~evaler)
         :else
         (.falseAlarmRate evaler)))

(defn false-negative-rate
  "False negative rate based on guesses so far Takes into account all known classes
  and outputs average fnr across all of them

  can be scoped down to a single class if class-label-idx supplied"
  [& {:keys [evaler class-label-idx edge-case]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))
           :edge-case (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(.falseNegativeRate ~evaler (int ~class-label-idx) (double ~edge-case))
         [{:evaler _
           :class-label-idx _
           :edge-case _}]
         (.falseNegativeRate evaler (int class-label-idx) edge-case)
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         `(.falseNegativeRate ~evaler (int ~class-label-idx))
         [{:evaler _
           :class-label-idx _}]
         (.falseNegativeRate evaler (int class-label-idx))
         [{:evaler (_ :guard seq?)}]
         `(.falseNegativeRate ~evaler)
         [{:evaler _}]
         (.falseNegativeRate evaler)))

(defn false-negatives
  "False negatives: correctly rejected"
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.falseNegatives ~evaler)
         :else
         (.falseNegatives evaler)))

(defn false-positive-rate
  "False positive rate based on guesses so far Takes into account all known classes
  and outputs average fpr across all of them

  can be scoped down to a single class if class-label-idx supplied"
  [& {:keys [evaler class-label-idx edge-case]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))
           :edge-case (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(.falsePositiveRate ~evaler (int ~class-label-idx) (double ~edge-case))
         [{:evaler _
           :class-label-idx _
           :edge-case _}]
         (.falsePositiveRate evaler (int class-label-idx) edge-case)
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         `(.falsePositiveRate ~evaler (int ~class-label-idx))
         [{:evaler _
           :class-label-idx _}]
         (.falsePositiveRate evaler (int class-label-idx))
         [{:evaler (_ :guard seq?)}]
         `(.falsePositiveRate ~evaler)
         [{:evaler _}]
         (.falsePositiveRate evaler)))

(defn false-positives
  "False positive: wrong guess"
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.falsePositives ~evaler)
         :else
         (.falsePositives evaler)))

(defn get-class-label
  "get the class a label is associated with given an idx"
  [& {:keys [evaler label-idx]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :label-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(.getClassLabel ~evaler (int ~label-idx))
         :else
         (.getClassLabel evaler (int label-idx))))

(defn get-confusion-matrix
  "Returns the confusion matrix variable"
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.getConfusionMatrix ~evaler)
         :else
         (.getConfusionMatrix evaler)))

(defn get-num-row-counter
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.getNumRowCounter ~evaler)
         :else
         (.getNumRowCounter evaler)))

(defn get-prediction-by-predicted-class
  "Get a list of predictions, for all data with the specified predicted class,
  regardless of the actual data class."
  [& {:keys [evaler idx-of-predicted-class]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :idx-of-predicted-class (:or (_ :guard number?)
                                        (_ :guard seq?))}]
         `(.getPredictionByPredictedClass ~evaler (int ~idx-of-predicted-class))
         :else
         (.getPredictionByPredictedClass evaler (int idx-of-predicted-class))))

(defn get-prediction-errors
  "Get a list of prediction errors, on a per-record basis"
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.getPredictionErrors ~evaler)
         :else
         (.getPredictionErrors evaler)))

(defn get-predictions
  "Get a list of predictions in the specified confusion matrix entry
  (i.e., for the given actua/predicted class pair)"
  [& {:keys [evaler actual-class-idx predicted-class-idx]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :actual-class-idx (:or (_ :guard number?)
                                  (_ :guard seq?))
           :predicted-class-idx (:or (_ :guard number?)
                                     (_ :guard seq?))}]
         `(.getPredictions ~evaler (int ~actual-class-idx) (int ~predicted-class-idx))
         :else
         (.getPredictions evaler actual-class-idx predicted-class-idx)))

(defn get-predictions-by-actual-class
  "Get a list of predictions, for all data with the specified actual class,
  regardless of the predicted class."
  [& {:keys [evaler actual-class-idx]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :actual-class-idx (:or (_ :guard number?)
                                  (_ :guard seq?))}]
         `(.getPredictionsByActualClass ~evaler (int ~actual-class-idx))
         :else
         (.getPredictionsByActualClass evaler actual-class-idx)))

(defn get-top-n-correct-count
  "Return the number of correct predictions according to top N value."
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.getTopNCorrectCount ~evaler)
         :else
         (.getTopNCorrectCount evaler)))

(defn get-top-n-total-count
  "Return the total number of top N evaluations."
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.getTopNTotalCount ~evaler)
         :else
         (.getTopNTotalCount evaler)))

(defn total-negatives
  "Total negatives true negatives + false negatives"
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.negative ~evaler)
         :else
         (.negative evaler)))

(defn total-positives
  "Returns all of the positive guesses: true positive + false negative"
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.positive ~evaler)
         :else
         (.positive evaler)))

(defn get-precision
  "Precision based on guesses so far Takes into account all known classes and
  outputs average precision across all of them.

  can be scoped to a label given its idx"
  [& {:keys [evaler class-label-idx edge-case]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))
           :edge-case (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(.precision ~evaler (int ~class-label-idx) (double ~edge-case))
         [{:evaler _
           :class-label-idx _
           :edge-case _}]
         (.precision evaler (int class-label-idx) edge-case)
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         `(.precision ~evaler (int ~class-label-idx))
         [{:evaler _
           :class-label-idx _}]
         (.precision evaler (int class-label-idx))
         [{:evaler (_ :guard seq?)}]
         `(.precision ~evaler)
         [{:evaler _}]
         (.precision evaler)))

(defn recall
  "Recall based on guesses so far Takes into account all known classes
  and outputs average recall across all of them

  can be scoped to a label given its idx"
  [& {:keys [evaler class-label-idx edge-case]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))
           :edge-case (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(.recall ~evaler (int ~class-label-idx) (double ~edge-case))
         [{:evaler _
           :class-label-idx _
           :edge-case _}]
         (.recall evaler (int class-label-idx) edge-case)
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         `(.recall ~evaler (int ~class-label-idx))
         [{:evaler _
           :class-label-idx _}]
         (.recall evaler (int class-label-idx))
         [{:evaler (_ :guard seq?)}]
         `(.recall ~evaler)
         [{:evaler _}]
         (.recall evaler)))

(defn top-n-accuracy
  "Top N accuracy of the predictions so far."
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.topNAccuracy ~evaler)
         :else
         (.topNAccuracy evaler)))

(defn true-negatives
  "True negatives: correctly rejected"
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.trueNegatives ~evaler)
         :else
         (.trueNegatives evaler)))

(defn true-positives
  "True positives: correctly rejected"
  [evaler]
  (match [evaler]
         [(_ :guard seq?)]
         `(.truePositives ~evaler)
         :else
         (.truePositives evaler)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; interact with a regression evaluator
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-mean-squared-error
  "returns the MSE"
  [& {:keys [regression-evaler column-idx]
      :as opts}]
  (match [opts]
         [{:regression-evaler (_ :guard seq?)
           :column-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         `(.meanSquaredError ~regression-evaler (int ~column-idx))
         :else
         (.meanSquaredError regression-evaler column-idx)))

(defn get-mean-absolute-error
  "returns MAE"
  [& {:keys [regression-evaler column-idx]
      :as opts}]
  (match [opts]
         [{:regression-evaler (_ :guard seq?)
           :column-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         `(.meanAbsoluteError ~regression-evaler (int ~column-idx))
         :else
         (.meanAbsoluteError regression-evaler column-idx)))

(defn get-root-mean-squared-error
  "returns rMSE"
  [& {:keys [regression-evaler column-idx]
      :as opts}]
  (match [opts]
         [{:regression-evaler (_ :guard seq?)
           :column-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         `(.rootMeanSquaredError ~regression-evaler (int ~column-idx))
         :else
         (.rootMeanSquaredError regression-evaler column-idx)))

(defn get-correlation-r2
  "return the R2 correlation"
  [& {:keys [regression-evaler column-idx]
      :as opts}]
  (match [opts]
         [{:regression-evaler (_ :guard seq?)
           :column-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         `(.correlationR2 ~regression-evaler (int ~column-idx))
         :else
         (.correlationR2 regression-evaler column-idx)))

(defn get-relative-squared-error
  "return relative squared error"
  [& {:keys [regression-evaler column-idx]
      :as opts}]
  (match [opts]
         [{:regression-evaler (_ :guard seq?)
           :column-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         `(.relativeSquaredError ~regression-evaler (int ~column-idx))
         :else
         (.relativeSquaredError regression-evaler column-idx)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; general
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn merge!
  "merges objects that implemented the IEvaluation interface

  evaler and other-evaler can be evaluations, ROCs or MultiClassRocs"
  [& {:keys [evaler other-evaler]
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :other-evaler (_ :guard seq?)}]
         `(doto ~evaler (.merge ~other-evaler))
         :else
         (doto evaler (.merge other-evaler))))
