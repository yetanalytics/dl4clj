(ns ^{:doc "ROC (Receiver Operating Characteristic) for binary classifiers, using the specified number of threshold steps.
implementation of the ROC class in dl4j. see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROC.html"}
    dl4clj.eval.roc
  (:import [org.deeplearning4j.eval ROC]))

(defn new-roc
  "creates a new ROC instance.  This implementation currently uses fixed
  steps of size 1.0 / thresholdSteps"
  [threshold-steps]
  (ROC. threshold-steps))

(defn calculate-area-under-curve
  "Calculate the AUC - Area Under Curve
  Utilizes trapezoidal integration internally"
  [roc]
  (.calculateAUC roc))

(defn eval!
  "Evaluate (collect statistics for) the given minibatch of data."
  [& {:keys [roc labels predictions mask-array]
      :as opts}]
  (if (contains? opts :mask-array)
    (doto roc (.eval labels predictions mask-array))
    (doto roc (.eval labels predictions))))

(defn eval-time-series!
  "Evaluate (collect stats for) the given minibatch of time series data"
  [& {:keys [roc labels network-predictions mask-array]
      :as opts}]
  (if (contains? opts :mask-array)
    (doto roc (.evalTimeSeries labels network-predictions mask-array))
    (doto roc (.evalTimeSeries labels network-predictions))))

(defn get-precision-recall-curve
  "returns the precision recall curve for the supplied ROC"
  [roc]
  (.getPrecisionRecallCurve roc))

(defn get-results
  "Get the ROC curve, as a set of points or as a
  set of (falsePositive, truePositive) points if as-array? is set to true"
  [& {:keys [roc as-array?]
      :or {as-array? false}
      :as opts}]
  (if (true? as-array?)
    (.getResultsAsArray roc)
    (.getResults roc)))

(defn merge!
  [& {:keys [roc other-roc]}]
  (doto roc (.merge other-roc)))
