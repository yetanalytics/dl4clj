(ns ^{:doc "ns for creating ROCs (Receiver Operating Characteristic) for binary and multi-class classifiers, using the specified number of threshold steps.

for multi-class classifiers, the ROC curves are produced by treating the predictions as a set of one-vs-all classifiers, and then calculating ROC curves for each. In practice, this means for N classes, we get N ROC curves.

for info on the binary ROCs, see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROC.html

for info on the multi-class ROCs, see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROCMultiClass.html

all fns in dl4clj.eval.interface.i-evaluation work with ROCs"}
    dl4clj.eval.roc.rocs
  (:import [org.deeplearning4j.eval ROCMultiClass ROC])
  (:require [dl4clj.utils :refer [generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multimethod for creating rocs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti rocs generic-dispatching-fn)

(defmethod rocs :multi-class [opts]
  (let [threshold (:threshold-steps (:multi-class opts))]
    (ROCMultiClass. threshold)))

(defmethod rocs :binary [opts]
  (let [threshold (:threshold-steps (:binary opts))]
    (ROC. threshold)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-binary-roc
  "creates a new ROC instance.

  This implementation currently uses fixed
  steps of size 1.0 / threshold-steps

  :threshold-steps (int), controls the step size for generating the ROC curve"
  [& {:keys [threshold-steps]
      :as opts}]
  (rocs {:binary opts}))

(defn new-multiclass-roc
  "creates a new ROC for multi-class classifiers

  This implementation currently uses fixed
  steps of size 1.0 / threshold-steps

  :threshold-steps (int), controls the step size for generating the ROC curve"
  [& {:keys [threshold-steps]
      :as opts}]
  (rocs {:multi-class opts}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; shared fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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

   - when false, return the curve as a set of pionts

 *******  need a better desc here, come back to this"
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
