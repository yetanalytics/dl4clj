(ns ^{:doc "namespace for creating optimization solvers.
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/solvers/package-summary.html"}
    dl4clj.optimize.solvers.optimizers
  (:import [org.deeplearning4j.optimize.solvers
            StochasticGradientDescent
            LineGradientDescent
            ConjugateGradient
            BaseOptimizer
            LBFGS
            BackTrackLineSearch])
  (:require [dl4clj.utils :refer [generic-dispatching-fn contains-many?]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method which creates an instance of the optimizer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti optimizers generic-dispatching-fn)

(defmethod optimizers :conjugate-geradient [opts]
  (let [conf (:conjugate-geradient opts)
        {nn-conf :nn-conf
         step-fn :step-fn
         listeners :listeners
         term-cond :termination-condition
         model :model} conf]
    (if (contains-many? conf :nn-conf :step-fn :listeners :termination-condition :model)
      (ConjugateGradient. nn-conf step-fn listeners term-cond model)
      (ConjugateGradient. nn-conf step-fn listeners model))))

(defmethod optimizers :lbfgs [opts]
  (let [conf (:lbfgs opts)
        {nn-conf :nn-conf
         step-fn :step-fn
         listeners :listeners
         term-cond :termination-condition
         model :model} conf]
    (if (contains-many? conf :nn-conf :step-fn :listeners :termination-condition :model)
      (LBFGS. nn-conf step-fn listeners term-cond model)
      (LBFGS. nn-conf step-fn listeners model))))

(defmethod optimizers :line-gradient-descent [opts]
  (let [conf (:line-gradient-descent opts)
        {nn-conf :nn-conf
         step-fn :step-fn
         listeners :listeners
         term-cond :termination-condition
         model :model} conf]
    (if (contains-many? conf :nn-conf :step-fn :listeners :termination-condition :model)
      (LineGradientDescent. nn-conf step-fn listeners term-cond model)
      (LineGradientDescent. nn-conf step-fn listeners model))))

(defmethod optimizers :stochastic-gradient-descent [opts]
  (let [conf (:stochastic-gradient-descent opts)
        {nn-conf :nn-conf
         step-fn :step-fn
         listeners :listeners
         term-cond :termination-condition
         model :model} conf]
    (if (contains-many? conf :nn-conf :step-fn :listeners :termination-condition :model)
      (StochasticGradientDescent. nn-conf step-fn listeners term-cond model)
      (StochasticGradientDescent. nn-conf step-fn listeners model))))

(defmethod optimizers :back-track-line-search [opts]
  (let [conf (:back-track-line-search opts)
        {model :model
         optimizer :optimizer
         step-fn :step-fn
         layer :layer} conf]
    (if (contains-many? conf :layer :step-fn :optimizer)
      (BackTrackLineSearch. layer step-fn optimizer)
      (BackTrackLineSearch. model optimizer))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns with arg descriptions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-conjugate-gradient-optimizer
  "creates a conjugate gradient optimizer

  :nn-conf (nn-conf), a neural network configuration
   - see: dl4clj.nn.conf.builders.nn-conf-builder

  :step-fn (step-fn), the step function to use
   - see: ... not yet fully implemented at time of writing this doc string

  :listeners (coll), collection of iteration listeners
   - see: dl4clj.optimize.listeners.listeners

  :termination-condition (keyword), reason to stop optimization
   - one of :esp, :norm2, :zero-direction
   - need to deal with the collection part of this********

  :model (model), a neural network model
   - see: dl4clj.nn.conf.builders.multi-layer-builders"
  [& {:keys [nn-conf step-fn listeners termination-condition model]
      :as opts}]
  (optimizers {:conjugate-geradient opts}))

(defn new-lbfgs-optimizer
  "creates a LBFGS optimizer

  :nn-conf (nn-conf), a neural network configuration
   - see: dl4clj.nn.conf.builders.nn-conf-builder

  :step-fn (step-fn), the step function to use
   - see: ... not yet fully implemented at time of writing this doc string

  :listeners (coll), collection of iteration listeners
   - see: dl4clj.optimize.listeners.listeners

  :termination-condition (keyword), reason to stop optimization
   - one of :esp, :norm2, :zero-direction
   - need to deal with the collection part of this********

  :model (model), a neural network model
   - see: dl4clj.nn.conf.builders.multi-layer-builders"
  [& {:keys [nn-conf step-fn listeners termination-condition model]
      :as opts}]
  (optimizers {:lbfgs opts}))

(defn new-line-gradient-descent-optimizer
  "creates a line gradient descent optimizer

  :nn-conf (nn-conf), a neural network configuration
   - see: dl4clj.nn.conf.builders.nn-conf-builder

  :step-fn (step-fn), the step function to use
   - see: ... not yet fully implemented at time of writing this doc string

  :listeners (coll), collection of iteration listeners
   - see: dl4clj.optimize.listeners.listeners

  :termination-condition (keyword), reason to stop optimization
   - one of :esp, :norm2, :zero-direction
   - need to deal with the collection part of this********

  :model (model), a neural network model
   - see: dl4clj.nn.conf.builders.multi-layer-builders"
  [& {:keys [nn-conf step-fn listeners termination-condition model]
      :as opts}]
  (optimizers {:line-gradient-descent opts}))

(defn new-stochastic-gradient-descent-optimizer
  "creates a stochastic gradient descent optimizer

  :nn-conf (nn-conf), a neural network configuration
   - see: dl4clj.nn.conf.builders.nn-conf-builder

  :step-fn (step-fn), the step function to use
   - see: ... not yet fully implemented at time of writing this doc string

  :listeners (coll), collection of iteration listeners
   - see: dl4clj.optimize.listeners.listeners

  :termination-condition (keyword), reason to stop optimization
   - one of :esp, :norm2, :zero-direction
   - need to deal with the collection part of this********

  :model (model), a neural network model
   - see: dl4clj.nn.conf.builders.multi-layer-builders"
  [& {:keys [nn-conf step-fn listeners termination-condition model]
      :as opts}]
  (optimizers {:stochastic-gradient-descent opts}))

(defn new-back-track-line-search-optimizer
  "creates a new back track line search optimizer

  :model (model), a neural network model
   - see: dl4clj.nn.conf.builders.multi-layer-builders

  :layer (layer), a layer within a neural network
   - see: dl4clj.nn.conf.builders.builders

  :optimizer (ConvexOptimizer) an existing optimizer that implements ConvexOptimizer

  :step-fn (step-fn), the step function to use
   - see: ... not yet fully implemented at time of writing this doc string"
  [& {:keys [model layer optimizer step-fn]
      :as opts}]
  (optimizers {:back-track-line-search opts}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Base optimizer methods not inherited from ConvexOptimizer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-default-step-fn-for-optimizer
  "returns the default step fn for a type of optimizer"
  [convex-optimizer-class]
  (.getDefaultStepFunctionForOptimizer convex-optimizer-class))

(defn get-iteration-count
  "get the number of iterations the model has been through
  -- I think, will need to test this assertion"
  [model]
  (.getIterationCount model))

(defn increment-iteration-count!
  "increments the iteration count for a model by the specified amount

  :increment-by (int), the specified amount

  returns the mutated model"
  [model increment-by]
  (doto model (.incrementIterationCount increment-by)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Back track line search methods
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-max-iterations
  "returns the max number of iterations for the optimizer"
  [back-track]
  (.getMaxIterations back-track))

(defn get-step-max
  "returns the max value of the step fn"
  [back-track]
  (.getStepMax back-track))

(defn set-abs-tolerance!
  "Sets the tolerance of absolute difference in function value

  :tolerance (double), the tolerance to set

  returns the back-track optimizer"
  [& {:keys [back-track tolerance]}]
  (doto back-track (.setAbsTolx tolerance)))

(defn set-max-iterations!
  "sets the max number of iterations for the optimizer

  :max-iterations (int), the value to set for the max iteration

  returns the back-track optimizer"
  [& {:keys [back-track max-iterations]}]
  (doto back-track (.setMaxIterations max-iterations)))

(defn set-relative-tolerance!
  "Sets the tolerance of relative difference in function value

  :tolerance (double), the tolerance to set

  returns the back-track optimizer"
  [& {:keys [back-track tolerance]}]
  (doto back-track (.setRelTolx tolerance)))

(defn set-score-for!
  "sets the score for the passed in params

  :params (INDArray), will need to test to write a good desc

  returns the back-track optimizer"
  [& {:keys [back-track params]}]
  (doto back-track (.setScoreFor params)))

(defn set-step-max!
  "sets the max step size for the back-track optimizer

  :step-max (double), the max value for the step size

  returns the back-track optimizer"
  [& {:keys [back-track step-max]}]
  (doto back-track (.setStepMax step-max)))
