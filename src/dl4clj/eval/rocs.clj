(ns ^{:doc "ns for creating ROCs (Receiver Operating Characteristic) for binary and multi-class classifiers, using the specified number of threshold steps.

for multi-class classifiers, the ROC curves are produced by treating the predictions as a set of one-vs-all classifiers, and then calculating ROC curves for each. In practice, this means for N classes, we get N ROC curves.

for info on the binary ROCs, see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROC.html

for info on the multi-class ROCs, see: https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROCMultiClass.html

all fns in dl4clj.eval.api.i-evaluation work with ROCs"}
    dl4clj.eval.rocs
  (:import [org.deeplearning4j.eval ROCMultiClass ROC])
  (:require [dl4clj.utils :refer [generic-dispatching-fn obj-or-code?]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multimethod for creating rocs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti rocs generic-dispatching-fn)

(defmethod rocs :multi-class [opts]
  (let [conf (:multi-class opts)
        threshold (:threshold-steps conf)]
    `(ROCMultiClass. ~threshold)))

(defmethod rocs :binary [opts]
  (let [conf (:binary opts)
        threshold (:threshold-steps conf)]
    `(ROC. ~threshold)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-binary-roc
  "creates a new ROC instance.

  This implementation currently uses fixed
  steps of size 1.0 / threshold-steps

  :as-code? (boolean), return the java object or the code for creating it

  threshold-steps (int), controls the step size for generating the ROC curve"
  [& {:keys [threshold-steps as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (rocs {:binary opts})]
    (obj-or-code? as-code? code)))

(defn new-multiclass-roc
  "creates a new ROC for multi-class classifiers

  This implementation currently uses fixed
  steps of size 1.0 / threshold-steps

  :as-code? (boolean), return the java object or the code for creating it

  threshold-steps (int), controls the step size for generating the ROC curve"
  [& {:keys [threshold-steps as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (rocs {:multi-class opts})]
    (obj-or-code? as-code? code)))
