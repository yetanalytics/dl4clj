(ns ^{:doc "step functions for optimizers
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/stepfunctions/package-summary.html"}
    dl4clj.optimize.step-functions.step-fns
  (:import [org.deeplearning4j.optimize.stepfunctions
            StepFunctions
            NegativeGradientStepFunction
            NegativeDefaultStepFunction
            GradientStepFunction
            StepFunctions
            DefaultStepFunction]
           [org.deeplearning4j.optimize.api StepFunction])
  (:require [dl4clj.nn.conf.step-fns :refer [step-fn]]
            [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Multi method for calling the step function constructors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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

  :step-fn (keyword), the step function
   - one of: :default-step-fn, :gradient-step-fn, :negative-default-step-fn
             :negative-gradient-step-fn"
  [& {:keys [nn-conf-step-fn]}]
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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; step from the step-function interface
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn step!
  "makes a step with the given params, the step-fn is returned

  :step-fn (step-fn), a step fn created by the fns in this ns

  :features (INDArray), the input data

  :lines (INDArray), the line.... need to figure out what this does

  :step (double), the size of the step to make

  :params (INDArray), the params...need to figure out exactly what this is

  :search-direction (INDArray), the line to step"
  [& {:keys [step-fn features line step params search-direction]
      :as opts}]
  (cond (contains-many? opts :params :search-direction :step)
        (doto step-fn (.step params search-direction step))
        (contains-many? opts :features :line :step)
        (doto step-fn (.step features line step))
        (contains-many? opts :features :line)
        (doto step-fn (.step features line))
        (contains? opts :step-fn)
        (doto step-fn (.step))
        :else
        (assert false "you must supply atleast a step function")))
