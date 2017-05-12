(ns dl4clj.nn.conf.step-fns
  (:import [org.deeplearning4j.nn.conf.stepfunctions
            StepFunction
            DefaultStepFunction
            GradientStepFunction
            NegativeDefaultStepFunction
            NegativeGradientStepFunction]))

(defmulti step-fn identity)

(defmethod step-fn :default-step-fn [opts]
  (DefaultStepFunction.))

(defmethod step-fn :gradient-step-fn [opts]
  (GradientStepFunction.))

(defmethod step-fn :negative-default-step-fn [opts]
  (NegativeDefaultStepFunction.))

(defmethod step-fn :negative-gradient-step-fn [opts]
  (NegativeGradientStepFunction.))
