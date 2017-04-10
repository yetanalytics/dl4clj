(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Model.html"}
  dl4clj.nn.api.model
  (:import [org.deeplearning4j.nn.api Model])
  (:require [dl4clj.nn.conf.utils :refer [contains-many?]]))

(defn accumulate-score
  "Sets a rolling tally for the score."
  [& {:keys [this accum]}]
  (.accumulateScore this (double accum)))

(defn apply-learning-rate-score-decay
  "Update learningRate using for this model."
  [^Model this]
  (.applyLearningRateScoreDecay this))

(defn batch-size
  "The current inputs batch size"
  [^Model this]
  (.batchSize this))

(defn clear
  "Clear input"
  [^Model this]
  (.clear this))

(defn compute-gradient-and-score
  "Update the score"
  [^Model this]
  (.computeGradientAndScore this))

(defn conf
  "The configuration for the neural network"
  [^Model this]
  (.conf this))

(defn fit
  "All models have a fit method"
  [& {:keys [this data]
      :as opts}]
  (cond-> this
    (contains? opts :data) (.fit data)
    :else
    .fit))

(defn get-optimizer
  "Returns this models optimizer"
  [^Model this]
  (.getOptimizer this))

(defn get-param
  "Get the parameter"
  [& {:keys [this param]}]
  (.getParam this (name param)))

(defn gradient
  "Calculate a gradient"
  [^Model this]
  (.gradient this))

(defn gradient-and-score
  "Get the gradient and score"
  [^Model this]
  (.gradientAndScore this))

(defn init
  [^Model this]
  (.init this))

(defn input
  "The input/feature matrix for the model"
  [^Model this]
  (.input this))

(defn iterate-once
  "Run one iteration"
  [& {:keys [this input]}]
  (.iterate this input))

(defn num-params
  "the number of parameters for the model"
  [& {:keys [this backwards?]
      :as opts}]
  (cond-> this
    (contains? opts :backwards?) (.numParams backwards?)
    :else
    .numParams))

(defn params
  "Parameters of the model (if any)"
  [^Model this]
  (.params this))

(defn param-table
  "The param table"
  [& {:keys [this backprop-params-only?]
      :as opts}]
  (cond-> this
    (contains? opts :backprop-params-only?) (.paramTable backprop-params-only?)
    :else
    .paramTable))

(defn score
  "The score for the model"
  [^Model this]
  (.score this))

(defn set-backprop-gradients-vew-array
  "Set the gradients array as a view of the full (backprop) network parameters
   NOTE: this is intended to be used internally in MultiLayerNetwork and ComputationGraph, not by users."
  [& {:keys [this gradients]}]
  (.setBackpropGradientsViewArray this gradients))

(defn set-conf
  "Setter for the configuration"
  [^Model this]
  (.setConf this conf))

(defn set-listeners
  "set the iteration listeners for the computational graph"
  [& {:keys [this listeners]}]
  (.setListeners this listeners))

(defn set-param
  "Set the parameter with a new ndarray"
  [& {:keys [this k v]}]
  (.setParam this k v))

(defn set-params
  "Set the parameters for this model."
  [& {:keys [this params]}]
  (.setParams this params))

(defn set-params-view-array
  "Set the initial parameters array as a view of the full (backprop) network parameters
   NOTE: this is intended to be used internally in MultiLayerNetwork and ComputationGraph, not by users."
  [& {:keys [this params]}]
  (.setParamsViewArray this params))

(defn set-param-table
  "Setter for the param table"
  [& {:keys [this param-table]}]
  (.setParamTable this param-table))

(defn updatez
  "Perform one update applying the gradient"
  [& {:keys [this gradient param-type]
      :as opts}]
  (if (contains? opts :param-type)
    (.update this gradient param-type)
    (.update this gradient)))
