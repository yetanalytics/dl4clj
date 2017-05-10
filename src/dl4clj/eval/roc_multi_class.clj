(ns ^{:doc "ROC (Receiver Operating Characteristic) for multi-class classifiers, using the specified number of threshold steps.

The ROC curves are produced by treating the predictions as a set of one-vs-all classifiers, and then calculating ROC curves for each. In practice, this means for N classes, we get N ROC curves.

see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROCMultiClass.html"}
    dl4clj.eval.roc-multi-class
  (:import [org.deeplearning4j.eval ROCMultiClass]))

(defn new-roc-multiclass
  "creates a new ROC for multi-class classifiers"
  [threshold-steps]
  (ROCMultiClass. threshold-steps))

(defn calculate-area-under-curve
  "Calculate the AUC - Area Under Curve
  Utilizes trapezoidal integration internally"
  [roc-mc]
  (.calculateAUC roc-mc))

(defn calculate-average-area-under-curve
  "Calculate the average (one-vs-all) AUC for all classes"
  [roc-mc]
  (.calculateAverageAUC roc-mc))

(defn eval!
  "Evaluate (collect statistics for) the given minibatch of data."
  [& {:keys [roc-mc labels predictions mask-array]
      :as opts}]
  (if (contains? opts :mask-array)
    (doto roc-mc (.eval labels predictions mask-array))
    (doto roc-mc (.eval labels predictions))))

(defn get-precision-recall-curve
  "returns the precision recall curve for the supplied ROC"
  [roc-mc]
  (.getPrecisionRecallCurve roc-mc))

(defn get-results
  "Get the ROC curve, as a set of points or as a
  set of (falsePositive, truePositive) points if as-array? is set to true"
  [& {:keys [roc-mc as-array?]
      :or {as-array? false}
      :as opts}]
  (if (true? as-array?)
    (.getResultsAsArray roc-mc)
    (.getResults roc-mc)))

(defn merge!
  [& {:keys [roc-mc other-roc-mc]}]
  (doto roc-mc (.merge other-roc-mc)))
