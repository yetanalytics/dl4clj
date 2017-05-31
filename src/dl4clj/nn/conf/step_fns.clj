(ns dl4clj.nn.conf.step-fns
  (:import [org.deeplearning4j.nn.conf.stepfunctions
            StepFunction
            DefaultStepFunction
            GradientStepFunction
            NegativeDefaultStepFunction
            NegativeGradientStepFunction]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti step-fn identity)

(defmethod step-fn :default-step-fn [opts]
  (DefaultStepFunction.))

(defmethod step-fn :gradient-step-fn [opts]
  (GradientStepFunction.))

(defmethod step-fn :negative-default-step-fn [opts]
  (NegativeDefaultStepFunction.))

(defmethod step-fn :negative-gradient-step-fn [opts]
  (NegativeGradientStepFunction.))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-default-step-fn
  "creates a new default step fn object"
  []
  (step-fn :default-step-fn))

(defn new-gradient-step-fn
  "creates a new gradient step fn object"
  []
  (step-fn :gradient-step-fn))

(defn new-negative-default-step-fn
  "creates a new negative default step fn object"
  []
  (step-fn :negative-default-step-fn))

(defn new-negative-gradient-step-fn
  "creates a new negative gradient step fn object"
  []
  (step-fn :negative-gradient-step-fn))
