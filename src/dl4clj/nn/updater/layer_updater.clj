(ns ^{:doc "implementation of the LayerUpdater class from dl4j.  Updates a layer

These fns happen behind the scene (when you fit a model)
- can use them to play around with a layer in combination with single forward passes

see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/updater/LayerUpdater.html"}
    dl4clj.nn.updater.layer-updater
  (:import [org.deeplearning4j.nn.updater LayerUpdater])
  (:require [dl4clj.constants :as enum]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn new-layer-updater
  "creates a new instance of the LayerUpdater class"
  []
  (LayerUpdater.))

(defn apply-lrate-decay-policy!
  "Update learning rate based on policy

  :decay-policy (keyword), the learning rate decay policy
   - one of :none, :exponential, :inverse, :poly, :sigmoid, :step, :torch-step :schedule :score

  :layer (layer), a neural network layer
   - see: dl4clj.nn.conf.builders.builders

  :iteration (int), the iteration to apply the policy

  :variable (str), the variable to apply the decay policy for

  returns a map of {:layer layer :updater updater}"
  [& {:keys [updater decay-policy layer iteration variable]}]
  (.applyLrDecayPolicy updater (enum/value-of {:learning-rate-policy decay-policy})
                       layer
                       iteration
                       variable)
  {:layer layer :updater updater})

(defn apply-momentum-decay-policy!
  "Updates current value of momentum based on the momentum schedule
   - if the momentum schedule exists

  :layer (layer), a neural network layer
   - see: dl4clj.nn.conf.builders.builders

  :iteration (int), the iteration to apply the policy

  :variable (str), the variable to apply the decay policy for

  returns a map of {:layer layer :updater updater}"
  [& {:keys [updater layer iteration variable]}]
  (.applyMomentumDecayPolicy updater layer iteration variable)
  {:layer layer :updater updater})

(defn pre-apply!
  "Apply gradient normalization: scale based on L2, clipping etc.

  :layer (layer), a neural network layer
   - see: dl4clj.nn.conf.builders.builders

  :gradient (gradient), the errors for the layer
   - see: dl4clj.nn.gradient.default-gradient

  :iteration (int), the iteration to apply the policy

  returns the updater"
  [& {:keys [updater layer gradient iteration]}]
  (.preApply updater layer gradient iteration)
  {:layer layer :updater updater})

(defn post-apply!
  "Apply the regularization

  :layer (layer), a neural network layer
   - see: dl4clj.nn.layers.layer-creation

  :gradient (INDArray or vec), the errors for the layer

  :param (str), the param to apply the gradient and regularization to

  :mini-batch-size (int), the size of the mini-batch

  returns the updater"
  [& {:keys [updater layer gradient param mini-batch-size]}]
  (.postApply updater layer (vec-or-matrix->indarray gradient) param mini-batch-size)
  {:layer layer :updater updater})

(defn get-updater-for-variable
  "returns a map of of {param-name gradient-updater}"
  [updater]
  (.getUpdaterForVariable updater))

;; all other fns found in the Updater interface
