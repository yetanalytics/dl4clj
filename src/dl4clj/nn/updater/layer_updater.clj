(ns ^{:doc ""}
    dl4clj.nn.updater.layer-updater
  (:import [org.deeplearning4j.nn.updater LayerUpdater])
  (:require [dl4clj.constants :as enum]))

(defn new-layer-updater
  "creates a new instance of the LayerUpdater class"
  []
  (LayerUpdater.))

(defn apply-learning-rate-decay-policy!
  "Update learning rate based on policy

  :decay-policy (keyword), the learning rate decay policy
   - one of :none, :exponential, :inverse, :poly, :sigmoid, :step, :torch-step :schedule :score

  :layer (layer), a neural network layer
   - see: dl4clj.nn.conf.builders.builders

  :iteration (int), the iteration to apply the policy

  :variable (str), the variable to apply the decay policy for

  returns the updater"
  [& {:keys [updater decay-policy layer iteration variable]}]
  (doto updater (.applyLrDecayPolicy (enum/value-of {:learning-rate-policy decay-policy})
                                     layer
                                     iteration
                                     variable)))

(defn apply-momentum-decay-policy!
  "Update momentum if schedule exist

  :layer (layer), a neural network layer
   - see: dl4clj.nn.conf.builders.builders

  :iteration (int), the iteration to apply the policy

  :variable (str), the variable to apply the decay policy for

  returns the updater"
  [& {:keys [updater layer iteration variable]}]
  (doto updater (.applyMomentumDecayPolicy layer iteration variable)))

(defn pre-apply!
  "Apply gradient normalization: scale based on L2, clipping etc.

  :layer (layer), a neural network layer
   - see: dl4clj.nn.conf.builders.builders

  :gradient (gradient), the errors for the layer
   - see: dl4clj.nn.gradient.default-gradient

  :iteration (int), the iteration to apply the policy

  returns the updater"
  [& {:keys [updater layer gradient iteration]}]
  (doto updater (.preApply layer gradient iteration)))

(defn post-apply!
  "Apply the regularization

  :layer (layer), a neural network layer
   - see: dl4clj.nn.conf.builders.builders

  :gradient (INDArray), the errors for the layer

  :param (str), the param to apply the gradient and regularization to

  :mini-batch-size (int), the size of the mini-batch

  returns the updater"
  [& {:keys [updater layer gradient-array param mini-batch-size]}]
  (doto updater (.postApply layer gradient-array param mini-batch-size)))

(defn get-updater-for-variable
  "returns a map of of {param-name gradient-updater}"
  [updater]
  (.getUpdaterForVariable updater))

;; all other fns found in the Updater interface
