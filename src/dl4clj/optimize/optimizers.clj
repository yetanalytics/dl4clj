(ns ^{:doc "namespace for creating optimization solvers.
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/solvers/package-summary.html"}
    dl4clj.optimize.optimizers
  (:import [org.deeplearning4j.optimize.solvers
            StochasticGradientDescent
            LineGradientDescent
            ConjugateGradient
            BaseOptimizer
            LBFGS
            BackTrackLineSearch])
  (:require [dl4clj.utils :refer [generic-dispatching-fn contains-many?]]))
;; dont think this is a user facing ns
;; will be removed in the core branch
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

  :step-fn (step-fn), the step function
   - one of: :default, :gradient, :negative-default
             :negative-gradient
   - for creating step-fns, see: dl4clj.optimize.step-functions.step-fns

  :listeners (coll), collection of iteration listeners
   - see: dl4clj.optimize.listeners.listeners

   - clojure data structures can be used here

  :termination-condition (coll), reason to stop optimization
   - one of :esp, :norm2, :zero-direction

   - for creating termination-conditions see: dl4clj.optimize.termination.terminations

   - clojure data structures can be used here

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network"
  [& {:keys [nn-conf step-fn listeners termination-condition model]
      :as opts}]
  (optimizers {:conjugate-geradient opts}))

(defn new-lbfgs-optimizer
  "creates a LBFGS optimizer

  :nn-conf (nn-conf), a neural network configuration
   - see: dl4clj.nn.conf.builders.nn-conf-builder

  :step-fn (step-fn), the step function
   - one of: :default, :gradient, :negative-default
             :negative-gradient
   - for creating step-fns, see: dl4clj.optimize.step-functions.step-fns

  :listeners (coll), collection of iteration listeners
   - see: dl4clj.optimize.listeners.listeners

   - clojure data structures can be used here

  :termination-condition (coll), reason to stop optimization
   - one of :esp, :norm2, :zero-direction

   - for creating termination-conditions see: dl4clj.optimize.termination.terminations

   - clojure data structures can be used here

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network"
  [& {:keys [nn-conf step-fn listeners termination-condition model]
      :as opts}]
  (optimizers {:lbfgs opts}))

(defn new-line-gradient-descent-optimizer
  "creates a line gradient descent optimizer

  :nn-conf (nn-conf), a neural network configuration
   - see: dl4clj.nn.conf.builders.nn-conf-builder

  :step-fn (step-fn), the step function
   - one of: :default, :gradient, :negative-default
             :negative-gradient
   - for creating step-fns, see: dl4clj.optimize.step-functions.step-fns

  :listeners (coll), collection of iteration listeners
   - see: dl4clj.optimize.listeners.listeners

   - clojure data structures can be used here

  :termination-condition (coll), reason to stop optimization
   - one of :esp, :norm2, :zero-direction

   - for creating termination-conditions see: dl4clj.optimize.termination.terminations

   - clojure data structures can be used here

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network"
  [& {:keys [nn-conf step-fn listeners termination-condition model]
      :as opts}]
  (optimizers {:line-gradient-descent opts}))

(defn new-stochastic-gradient-descent-optimizer
  "creates a stochastic gradient descent optimizer

  :nn-conf (nn-conf), a neural network configuration
   - see: dl4clj.nn.conf.builders.nn-conf-builder

  :step-fn (step-fn), the step function
   - one of: :default, :gradient, :negative-default
             :negative-gradient
   - for creating step-fns, see: dl4clj.optimize.step-functions.step-fns

  :listeners (coll), collection of iteration listeners
   - see: dl4clj.optimize.listeners.listeners

  :termination-condition (coll), reason to stop optimization
   - one of :esp, :norm2, :zero-direction

   - for creating termination-conditions see: dl4clj.optimize.termination.terminations

   - clojure data structures can be used here

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network"
  [& {:keys [nn-conf step-fn listeners termination-condition model]
      :as opts}]
  (optimizers {:stochastic-gradient-descent opts}))

(defn new-back-track-line-search-optimizer
  "creates a new back track line search optimizer

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network

  :layer (layer), a layer within a neural network
   - see: dl4clj.nn.conf.builders.builders

  :optimizer (ConvexOptimizer) an existing optimizer that implements ConvexOptimizer

  :step-fn (step-fn), the step function
   - one of: :default, :gradient, :negative-default
             :negative-gradient
   - for creating step-fns, see: dl4clj.optimize.step-functions.step-fns"
  [& {:keys [model layer optimizer step-fn]
      :as opts}]
  (optimizers {:back-track-line-search opts}))
