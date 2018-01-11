(ns dl4clj.eval.api.eval
  (:import [org.deeplearning4j.eval Evaluation RegressionEvaluation BaseEvaluation
            IEvaluation])
  (:require [dl4clj.utils :refer [contains-many? get-labels obj-or-code? eval-if-code]]
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

  :record-meta-data (coll) meta data that extends java.io.Serializable"
  [& {:keys [labels network-predictions mask-array record-meta-data evaler
             mln features predicted-idx actual-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :network-predictions (:or (_ :guard vector?)
                                     (_ :guard seq?))
           :record-meta-data (_ :guard seq?)}]
         (obj-or-code?
          as-code?
          `(doto ~evaler
             (.eval (vec-or-matrix->indarray ~labels)
                    (vec-or-matrix->indarray ~network-predictions)
                    (into '() ~record-meta-data))))
         [{:evaler _
           :labels _
           :network-predictions _
           :record-meta-data _}]
         (let [[e-obj l-vec p-vec md-obj] (eval-if-code [evaler seq?]
                                                        [labels seq?]
                                                        [network-predictions seq?]
                                                        [record-meta-data seq?])]
           (doto e-obj
             (.eval (vec-or-matrix->indarray l-vec)
                   (vec-or-matrix->indarray p-vec)
                   (into '() md-obj))))
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :network-predictions (:or (_ :guard vector?)
                                     (_ :guard seq?))
           :mask-array (:or (_ :guard vector?)
                            (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~evaler
           (.eval (vec-or-matrix->indarray ~labels)
                  (vec-or-matrix->indarray ~network-predictions)
                  (vec-or-matrix->indarray ~mask-array))))
         [{:evaler _
           :labels _
           :network-predictions _
           :mask-array _}]
         (let [[e-obj l-vec p-vec m-vec] (eval-if-code [evaler seq?]
                                                       [labels seq?]
                                                       [network-predictions seq?]
                                                       [mask-array seq?])]
           (doto e-obj
             (.eval (vec-or-matrix->indarray l-vec)
                    (vec-or-matrix->indarray p-vec)
                    (vec-or-matrix->indarray m-vec))))
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))
           :mln (_ :guard seq?)}]
         (obj-or-code?
          as-code?
          `(doto ~evaler (.eval (vec-or-matrix->indarray ~labels)
                               (vec-or-matrix->indarray ~features)
                               ~mln)))
         [{:evaler _
           :labels _
           :features _
           :mln _}]
         (let [[e-obj l-vec f-vec mln-obj] (eval-if-code [evaler seq?]
                                                         [labels seq?]
                                                         [features seq?]
                                                         [mln seq?])]
           (doto e-obj (.eval (vec-or-matrix->indarray l-vec)
                              (vec-or-matrix->indarray f-vec)
                              mln-obj)))
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :network-predictions (:or (_ :guard vector?)
                                     (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~evaler
            (.eval (vec-or-matrix->indarray ~labels)
                   (vec-or-matrix->indarray ~network-predictions))))
         [{:evaler _
           :labels _
           :network-predictions _}]
         (let [[e-obj l-vec p-vec] (eval-if-code [evaler seq?]
                                                 [labels seq?]
                                                 [network-predictions seq?])]
           (doto e-obj
             (.eval (vec-or-matrix->indarray l-vec)
                    (vec-or-matrix->indarray p-vec))))
         [{:evaler (_ :guard seq?)
           :predicted-idx (:or (_ :guard number?)
                               (_ :guard seq?))
           :actual-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~evaler (.eval (int ~predicted-idx) (int ~actual-idx))))
         [{:evaler _
           :predicted-idx _
           :actual-idx _}]
         (let [[e-obj p-idx a-idx] (eval-if-code [evaler seq?]
                                                 [predicted-idx seq?]
                                                 [actual-idx seq? number?])]
           (doto e-obj (.eval p-idx a-idx)))))

(defn eval-time-series!
  "evalatues a time series given labels and predictions.

  labels-mask is optional and only applies when there is a mask"
  [& {:keys [labels predicted labels-mask evaler as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :predicted (:or (_ :guard vector?)
                           (_ :guard seq?))
           :labels-mask (:or (_ :guard vector?)
                             (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~evaler (.evalTimeSeries
                          (vec-or-matrix->indarray ~labels)
                          (vec-or-matrix->indarray ~predicted)
                          (vec-or-matrix->indarray ~labels-mask))))
         [{:evaler _
           :labels _
           :predicted _
           :labels-mask _}]
         (let [[e-obj l-vec p-vec m-vec] (eval-if-code [evaler seq?]
                                                       [labels seq?]
                                                       [predicted seq?]
                                                       [labels-mask seq?])]
           (doto e-obj (.evalTimeSeries
                        (vec-or-matrix->indarray l-vec)
                        (vec-or-matrix->indarray p-vec)
                        (vec-or-matrix->indarray m-vec))))
         [{:evaler (_ :guard seq?)
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :predicted (:or (_ :guard vector?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~evaler (.evalTimeSeries
                          (vec-or-matrix->indarray ~labels)
                          (vec-or-matrix->indarray ~predicted))))
         [{:evaler _
           :labels _
           :predicted _}]
         (let [[e-obj l-vec p-vec] (eval-if-code [evaler seq?]
                                                 [labels seq?]
                                                 [predicted seq?])]
           (doto e-obj (.evalTimeSeries
                        (vec-or-matrix->indarray l-vec)
                        (vec-or-matrix->indarray p-vec))))))

(defn get-stats
  "Method to obtain the classification report as a String"
  [& {:keys [evaler suppress-warnings? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :suppress-warnings? (:or (_ :guard boolean?)
                                    (_ :guard seq?))}]
         (obj-or-code? as-code? `(.stats ~evaler ~suppress-warnings?))
         [{:evaler _
           :suppress-warnings? _}]
         (let [[w-b] (eval-if-code [suppress-warnings? seq? boolean?])]
           (.stats evaler w-b))
         [{:evaler (_ :guard seq?)}]
         (obj-or-code? as-code? `(.stats ~evaler))
         [{:evaler _}]
         (.stats evaler)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; classification evaluator interaction fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-accuracy
  "Accuracy: (TP + TN) / (P + N)"
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.accuracy ~evaler))
         :else
         (.accuracy evaler)))

(defn class-count
  "Returns the number of times the given label has actually occurred"
  [& {:keys [evaler class-label-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         (obj-or-code? as-code? `(.classCount ~evaler (int ~class-label-idx)))
         :else
         (let [[idx-n] (eval-if-code [class-label-idx seq? number?])]
           (.classCount evaler (int idx-n)))))

(defn confusion-to-string
  "Get a String representation of the confusion matrix"
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.confusionToString ~evaler))
         :else
         (.confusionToString evaler)))

(defn f1
  "TP: true positive FP: False Positive FN: False Negative
  F1 score = 2 * TP / (2TP + FP + FN),

  the calculation will only be done for a single class if that classes idx is supplied
   -here class refers to the labels the network was trained on"
  [& {:keys [evaler class-label-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         (obj-or-code? as-code? `(.f1 ~evaler (int ~class-label-idx)))
         [{:evaler _
           :class-label-idx _ }]
         (let [[idx-n] (eval-if-code [class-label-idx seq? number?])]
           (.f1 evaler (int idx-n)))
         [{:evaler (_ :guard seq?)}]
         (obj-or-code? as-code? `(.f1 ~evaler))
         [{:evaler _}]
         (.f1 evaler)))

(defn false-alarm-rate
  "False Alarm Rate (FAR) reflects rate of misclassified to classified records"
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.falseAlarmRate ~evaler))
         :else
         (.falseAlarmRate evaler)))

(defn false-negative-rate
  "False negative rate based on guesses so far Takes into account all known classes
  and outputs average fnr across all of them

  can be scoped down to a single class if class-label-idx supplied"
  [& {:keys [evaler class-label-idx edge-case as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))
           :edge-case (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.falseNegativeRate ~evaler (int ~class-label-idx) (double ~edge-case)))
         [{:evaler _
           :class-label-idx _
           :edge-case _}]
         (let [[idx-n edge-case-n] (eval-if-code [class-label-idx seq? number?]
                                                 [edge-case seq? number?])]
           (.falseNegativeRate evaler (int idx-n) (double edge-case-n)))
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         (obj-or-code? as-code? `(.falseNegativeRate ~evaler (int ~class-label-idx)))
         [{:evaler _
           :class-label-idx _}]
         (let [[idx-n] (eval-if-code [class-label-idx seq? number?])]
           (.falseNegativeRate evaler (int idx-n)))
         [{:evaler (_ :guard seq?)}]
         (obj-or-code? as-code? `(.falseNegativeRate ~evaler))
         [{:evaler _}]
         (.falseNegativeRate evaler)))

(defn false-negatives
  "False negatives: correctly rejected"
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.falseNegatives ~evaler))
         :else
         (.falseNegatives evaler)))

(defn false-positive-rate
  "False positive rate based on guesses so far Takes into account all known classes
  and outputs average fpr across all of them

  can be scoped down to a single class if class-label-idx supplied"
  [& {:keys [evaler class-label-idx edge-case as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))
           :edge-case (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.falsePositiveRate ~evaler (int ~class-label-idx) (double ~edge-case)))
         [{:evaler _
           :class-label-idx _
           :edge-case _}]
         (let [[idx-n edge-case-n] (eval-if-code [class-label-idx seq? number?]
                                                 [edge-case seq? number?])]
           (.falsePositiveRate evaler (int idx-n) edge-case-n))
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.falsePositiveRate ~evaler (int ~class-label-idx)))
         [{:evaler _
           :class-label-idx _}]
         (let [[idx-n] (eval-if-code [class-label-idx seq? number?])]
           (.falsePositiveRate evaler (int idx-n)))
         [{:evaler (_ :guard seq?)}]
         (obj-or-code?
          as-code?
          `(.falsePositiveRate ~evaler))
         [{:evaler _}]
         (.falsePositiveRate evaler)))

(defn false-positives
  "False positive: wrong guess"
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.falsePositives ~evaler))
         :else
         (.falsePositives evaler)))

(defn get-class-label
  "get the class a label is associated with given an idx"
  [& {:keys [evaler label-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :label-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.getClassLabel ~evaler (int ~label-idx)))
         :else
         (let [[l-idx] (eval-if-code [label-idx seq? number?])]
           (.getClassLabel evaler (int l-idx)))))

(defn get-confusion-matrix
  "Returns the confusion matrix variable"
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getConfusionMatrix ~evaler))
         :else
         (.getConfusionMatrix evaler)))

(defn get-num-row-counter
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getNumRowCounter ~evaler))
         :else
         (.getNumRowCounter evaler)))

(defn get-prediction-by-predicted-class
  "Get a list of predictions, for all data with the specified predicted class,
  regardless of the actual data class."
  [& {:keys [evaler idx-of-predicted-class as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :idx-of-predicted-class (:or (_ :guard number?)
                                        (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.getPredictionByPredictedClass ~evaler (int ~idx-of-predicted-class)))
         :else
         (let [[idx-n] (eval-if-code [idx-of-predicted-class seq? number?])]
           (.getPredictionByPredictedClass evaler (int idx-n)))))

(defn get-prediction-errors
  "Get a list of prediction errors, on a per-record basis"
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getPredictionErrors ~evaler))
         :else
         (.getPredictionErrors evaler)))

(defn get-predictions
  "Get a list of predictions in the specified confusion matrix entry
  (i.e., for the given actua/predicted class pair)"
  [& {:keys [evaler actual-class-idx predicted-class-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :actual-class-idx (:or (_ :guard number?)
                                  (_ :guard seq?))
           :predicted-class-idx (:or (_ :guard number?)
                                     (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.getPredictions ~evaler (int ~actual-class-idx) (int ~predicted-class-idx)))
         :else
         (let [[a-idx p-idx] (eval-if-code [actual-class-idx seq? number?]
                                           [predicted-class-idx seq? number?])]
           (.getPredictions evaler a-idx p-idx))))

(defn get-predictions-by-actual-class
  "Get a list of predictions, for all data with the specified actual class,
  regardless of the predicted class."
  [& {:keys [evaler actual-class-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :actual-class-idx (:or (_ :guard number?)
                                  (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.getPredictionsByActualClass ~evaler (int ~actual-class-idx)))
         :else
         (let [[a-idx] (eval-if-code [actual-class-idx seq? number?])]
           (.getPredictionsByActualClass evaler a-idx))))

(defn get-top-n-correct-count
  "Return the number of correct predictions according to top N value."
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getTopNCorrectCount ~evaler))
         :else
         (.getTopNCorrectCount evaler)))

(defn get-top-n-total-count
  "Return the total number of top N evaluations."
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getTopNTotalCount ~evaler))
         :else
         (.getTopNTotalCount evaler)))

(defn total-negatives
  "Total negatives true negatives + false negatives"
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.negative ~evaler))
         :else
         (.negative evaler)))

(defn total-positives
  "Returns all of the positive guesses: true positive + false negative"
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.positive ~evaler))
         :else
         (.positive evaler)))

(defn get-precision
  "Precision based on guesses so far Takes into account all known classes and
  outputs average precision across all of them.

  can be scoped to a label given its idx"
  [& {:keys [evaler class-label-idx edge-case as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))
           :edge-case (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.precision ~evaler (int ~class-label-idx) (double ~edge-case)))
         [{:evaler _
           :class-label-idx _
           :edge-case _}]
         (let [[l-idx edge-case-n] (eval-if-code [class-label-idx seq? number?]
                                                 [edge-case seq? number?])]
           (.precision evaler (int l-idx) edge-case-n))
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.precision ~evaler (int ~class-label-idx)))
         [{:evaler _
           :class-label-idx _}]
         (let [[l-idx] (eval-if-code [class-label-idx seq?])]
           (.precision evaler (int l-idx)))
         [{:evaler (_ :guard seq?)}]
         (obj-or-code? as-code? `(.precision ~evaler))
         [{:evaler _}]
         (.precision evaler)))

(defn recall
  "Recall based on guesses so far Takes into account all known classes
  and outputs average recall across all of them

  can be scoped to a label given its idx"
  [& {:keys [evaler class-label-idx edge-case as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))
           :edge-case (:or (_ :guard number?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.recall ~evaler (int ~class-label-idx) (double ~edge-case)))
         [{:evaler _
           :class-label-idx _
           :edge-case _}]
         (let [[l-idx edge-case-n] (eval-if-code [class-label-idx seq? number?]
                                                 [edge-case seq? number?])]
           (.recall evaler (int l-idx) edge-case-n))
         [{:evaler (_ :guard seq?)
           :class-label-idx (:or (_ :guard number?)
                                 (_ :guard seq?))}]
         (obj-or-code? as-code? `(.recall ~evaler (int ~class-label-idx)))
         [{:evaler _
           :class-label-idx _}]
         (let [[l-idx] (eval-if-code [class-label-idx seq? number?])]
           (.recall evaler (int l-idx)))
         [{:evaler (_ :guard seq?)}]
         (obj-or-code? as-code? `(.recall ~evaler))
         [{:evaler _}]
         (.recall evaler)))

(defn top-n-accuracy
  "Top N accuracy of the predictions so far."
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.topNAccuracy ~evaler))
         :else
         (.topNAccuracy evaler)))

(defn true-negatives
  "True negatives: correctly rejected"
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.trueNegatives ~evaler))
         :else
         (.trueNegatives evaler)))

(defn true-positives
  "True positives: correctly rejected"
  [evaler & {:keys [as-code?]
             :or {as-code? true}}]
  (match [evaler]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.truePositives ~evaler))
         :else
         (.truePositives evaler)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; interact with a regression evaluator
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-mean-squared-error
  "returns the MSE"
  [& {:keys [regression-evaler column-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:regression-evaler (_ :guard seq?)
           :column-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.meanSquaredError ~regression-evaler (int ~column-idx)))
         :else
         (let [[idx-n] (eval-if-code [column-idx seq? number?])]
           (.meanSquaredError regression-evaler idx-n))))

(defn get-mean-absolute-error
  "returns MAE"
  [& {:keys [regression-evaler column-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:regression-evaler (_ :guard seq?)
           :column-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.meanAbsoluteError ~regression-evaler (int ~column-idx)))
         :else
         (let [[idx-n] (eval-if-code [column-idx seq? number?])]
           (.meanAbsoluteError regression-evaler idx-n))))

(defn get-root-mean-squared-error
  "returns rMSE"
  [& {:keys [regression-evaler column-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:regression-evaler (_ :guard seq?)
           :column-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.rootMeanSquaredError ~regression-evaler (int ~column-idx)))
         :else
         (let [[idx-n] (eval-if-code [column-idx seq? number?])]
           (.rootMeanSquaredError regression-evaler idx-n))))

(defn get-correlation-r2
  "return the R2 correlation"
  [& {:keys [regression-evaler column-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:regression-evaler (_ :guard seq?)
           :column-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.correlationR2 ~regression-evaler (int ~column-idx)))
         :else
         (let [[idx-n] (eval-if-code [column-idx seq? number?])]
           (.correlationR2 regression-evaler idx-n))))

(defn get-relative-squared-error
  "return relative squared error"
  [& {:keys [regression-evaler column-idx as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:regression-evaler (_ :guard seq?)
           :column-idx (:or (_ :guard number?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.relativeSquaredError ~regression-evaler (int ~column-idx)))
         :else
         (let [[idx-n] (eval-if-code [column-idx seq? number?])]
           (.relativeSquaredError regression-evaler idx-n))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; general
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn merge!
  "merges objects that implemented the IEvaluation interface

  evaler and other-evaler can be evaluations, ROCs or MultiClassRocs"
  [& {:keys [evaler other-evaler as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:evaler (_ :guard seq?)
           :other-evaler (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~evaler (.merge ~other-evaler)))
         :else
         (let [[e1 e2] (eval-if-code [evaler seq?]
                                     [other-evaler seq?])]
           (doto e1 (.merge e2)))))
