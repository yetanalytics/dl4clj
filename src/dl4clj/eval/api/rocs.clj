(ns dl4clj.eval.api.rocs
  (:import [org.deeplearning4j.eval ROCMultiClass ROC])
  (:require [clojure.core.match :refer [match]]))

(defn calculate-area-under-curve
  "Calculate the AUC - Area Under Curve
  Utilizes trapezoidal integration internally

  :roc (roc) either a binary or multi-class roc

  :class-idx (int), the index of the class you care about
   - shoud only be supplied when the roc is a multi-class roc"
  [& {:keys [roc class-idx]
      :as opts}]
  (match [opts]
         [{:roc (_ :guard seq?)
           :class-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(.calculateAUC ~roc (int ~class-idx))
         [{:roc _
           :class-idx _}]
         (.calculateAUC roc class-idx)
         [{:roc (_ :guard seq?)}]
         `(.calculateAUC ~roc)
         [{:roc _}]
         (.calculateAUC roc)))

(defn get-precision-recall-curve
  "returns the precision recall curve for the supplied ROC

  :roc (roc) either a binary or multi-class roc

  :class-idx (int), the index of the class you care about
   - shoud only be supplied when the roc is a multi-class roc"
  [& {:keys [roc class-idx]
      :as opts}]
  (match [opts]
         [{:roc (_ :guard seq?)
           :class-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(.getPrecisionRecallCurve ~roc (int ~class-idx))
         [{:roc _
           :class-idx _}]
         (.getPrecisionRecallCurve roc class-idx)
         [{:roc (_ :guard seq?)}]
         `(.getPrecisionRecallCurve ~roc)
         [{:roc _}]
         (.getPrecisionRecallCurve roc)))

(defn get-results
  "Get the ROC curve

  :roc (roc), either a binary or multi-class roc

  :class-idx (int), the index of the class you care about
   - shoud only be supplied when the roc is a multi-class roc

  returns the curve as a set of pionts"
  [& {:keys [roc class-idx]
      :as opts}]
  (match [opts]
         [{:roc (_ :guard seq?)
           :class-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(.getResults ~roc (int ~class-idx))
         [{:roc _
           :class-idx _}]
         (.getResults roc class-idx)
         [{:roc (_ :guard seq?)}]
         `(.getResults ~roc)
         [{:roc _}]
         (.getResults roc)))

(defn get-results-as-array
  "Get the ROC curve

  :roc (roc), either a binary or multi-class roc

  :class-idx (int), the index of the class you care about
   - shoud only be supplied when the roc is a multi-class roc

  returns the curve as a set of (false-positive, true-positive) points"
  [& {:keys [roc class-idx]
      :as opts}]
  (match [opts]
         [{:roc (_ :guard seq?)
           :class-idx (:or (_ :guard number?)
                           (_ :guard seq?))}]
         `(.getResultsAsArray ~roc (int ~class-idx))
         [{:roc _
           :class-idx _}]
         (.getResultsAsArray roc class-idx)
         [{:roc (_ :guard seq?)}]
         `(.getResultsAsArray ~roc)
         [{:roc _}]
         (.getResultsAsArray roc)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; only for multi-class ROCs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn calculate-average-area-under-curve
  "Calculate the average (one-vs-all) AUC for all classes"
  [roc-mc]
  (match [roc-mc]
         [(_ :guard seq?)]
         `(.calculateAverageAUC ~roc-mc)
         :else
         (.calculateAverageAUC roc-mc)))
