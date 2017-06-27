(ns dl4clj.eval.api.rocs
  (:import [org.deeplearning4j.eval ROCMultiClass ROC]))

(defn calculate-area-under-curve
  "Calculate the AUC - Area Under Curve
  Utilizes trapezoidal integration internally

  :roc (roc) either a binary or multi-class roc

  :class-idx (int), the index of the class you care about
   - shoud only be supplied when the roc is a multi-class roc"
  [& {:keys [roc class-idx]
      :as opts}]
  (if (contains? opts :class-idx)
    (.calculateAUC roc class-idx)
    (.calculateAUC roc)))

(defn get-precision-recall-curve
  "returns the precision recall curve for the supplied ROC

  :roc (roc) either a binary or multi-class roc

  :class-idx (int), the index of the class you care about
   - shoud only be supplied when the roc is a multi-class roc"
  [& {:keys [roc class-idx]
      :as opts}]
  (if (contains? opts :class-idx)
    (.getPrecisionRecallCurve roc class-idx)
    (.getPrecisionRecallCurve roc)))

(defn get-results
  "Get the ROC curve

  :roc (roc), either a binary or multi-class roc

  :class-idx (int), the index of the class you care about
   - shoud only be supplied when the roc is a multi-class roc

  :as-array? (boolean), defaults to false

   - when true, return the curve as a set of
   (false-positive, true-positive) points

   - when false, return the curve as a set of pionts"
  [& {:keys [roc as-array? class-idx]
      :or {as-array? false}
      :as opts}]
  (if (true? as-array?)
    (.getResultsAsArray roc)
    (.getResults roc)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; only for multi-class ROCs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn calculate-average-area-under-curve
  "Calculate the average (one-vs-all) AUC for all classes"
  [roc-mc]
  (.calculateAverageAUC roc-mc))
