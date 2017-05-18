(ns ^{:doc "an extension of IterationListener that adds onEpochStart, onEpochEnd, onForwardPass and onBackwardPass methods
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/TrainingListener.html"}
    dl4clj.optimize.api.training-listener
  (:import [org.deeplearning4j.optimize.api TrainingListener]))


;; depending on testing, may want to return the model instead of the listener

(defn on-backward-pass!
  "Called once per iteration, after gradients have been calculated and updated gradients
  can be returned by calling (gradient model)

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network

  returns the listener"
  [& {:keys [listener model]}]
  (doto listener (.onBackwardPass model)))

(defn on-epoch-end!
  "Called once at the end of each epoch, when fitting a model

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network

  returns the listener"
  [& {:keys [listener model]}]
  (doto listener (.onEpochEnd model)))

(defn on-epoch-start!
  "Called once at the start of each epochm when fitting a model

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network

  returns the listener"
  [& {:keys [listener model]}]
  (doto listener (.onEpochStart model)))

(defn on-forward-pass!
  "called once per iteration, for activations during training time

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network

  :activations (list (INDArray)) the activation values

  returns the listener"
  [& {:keys [listener model activations]}]
  (doto listener (.onForwardPass model activations)))

(defn on-gradient-calc!
  "called once per iteration before the gradients are updated.

  Note that gradients will likely be updated in-place
  - thus they should be copied or processed synchronously in this method.

  get the gradients by passing the model to (gradient)

  :model (model), A Model is meant for predicting something from data.
   - either a nn-layer or a multi-layer-network

  returns the listener"
  [& {:keys [listener model]}]
  (doto listener (.onGradientCalculation model)))
