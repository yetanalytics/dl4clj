(ns ^{:doc "step functions for optimizers
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/stepfunctions/package-summary.html"}
    dl4clj.optimize.step-functions.step-fns
  (:import [org.deeplearning4j.optimize.stepfunctions
            StepFunctions
            NegativeGradientStepFunction
            NegativeDefaultStepFunction
            GradientStepFunction
            StepFunctions
            DefaultStepFunction])
  (:require [dl4clj.nn.conf.step-fns :refer [step-fn]]
            [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Multi method for calling the step function constructors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; not user facing, will be removed

(defmulti step-fns identity)

(defmethod step-fns :default [opts]
  (DefaultStepFunction.))

(defmethod step-fns :gradient [opts]
  (GradientStepFunction.))

(defmethod step-fns :negative-default [opts]
  (NegativeDefaultStepFunction.))

(defmethod step-fns :negative-gradient [opts]
  (NegativeGradientStepFunction.))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns for creating step fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-step-fn-from-nn-conf-step-fn
  "creates a step fn from those specified in dl4clj.nn.conf.step-fns

  step-fn (keyword), the step function
   - one of: :default-step-fn, :gradient-step-fn, :negative-default-step-fn
             :negative-gradient-step-fn"
  [nn-conf-step-fn]
  (StepFunctions/createStepFunction (step-fn nn-conf-step-fn)))

(defn new-default-step-fn
  "creates a new default step function instance"
  []
  (step-fns :default))

(defn new-gradient-step-fn
  "creates a new gradient step function instance"
  []
  (step-fns :gradient))

(defn new-negative-default-step-fn
  "creates a new negative default step function instance"
  []
  (step-fns :negative-default))

(defn new-negative-gradient-step-fn
  "creates a new negative gradient step function instance"
  []
  (step-fns :negative-gradient))
