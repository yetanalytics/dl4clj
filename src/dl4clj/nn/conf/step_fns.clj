(ns dl4clj.nn.conf.step-fns
  (:import [org.deeplearning4j.nn.conf.stepfunctions
            StepFunction
            DefaultStepFunction
            GradientStepFunction
            NegativeDefaultStepFunction
            NegativeGradientStepFunction]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder])
  (:require [dl4clj.utils :refer [obj-or-code?]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti step-fn identity)

(defmethod step-fn :default-step-fn [opts]
  `(DefaultStepFunction.))

(defmethod step-fn :gradient-step-fn [opts]
  `(GradientStepFunction.))

(defmethod step-fn :negative-default-step-fn [opts]
  `(NegativeDefaultStepFunction.))

(defmethod step-fn :negative-gradient-step-fn [opts]
  `(NegativeGradientStepFunction.))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-default-step-fn
  "creates a new default step fn object"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (step-fn :default-step-fn)]
    (obj-or-code? as-code? code)))

(defn new-gradient-step-fn
  "creates a new gradient step fn object"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (step-fn :gradient-step-fn)]
    (obj-or-code? as-code? code)))

(defn new-negative-default-step-fn
  "creates a new negative default step fn object"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (step-fn :negative-default-step-fn)]
    (obj-or-code? as-code? code)))

(defn new-negative-gradient-step-fn
  "creates a new negative gradient step fn object"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (step-fn :negative-gradient-step-fn)]
    (obj-or-code? as-code? code)))
