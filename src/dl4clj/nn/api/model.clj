(ns ^{:doc "A Model is meant for predicting something from data.
 Note that this is not like supervised learning where there are labels attached to the examples.
 see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Model.html"}
  dl4clj.nn.api.model
  (:import [org.deeplearning4j.nn.api Model])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn accumulate-score!
  "Sets a rolling tally for the score."
  [& {:keys [model accum]}]
  (doto model
    (.accumulateScore accum)))

(defn apply-learning-rate-score-decay!
  "Update learningRate using for this model."
  [model]
  (doto model
    (.applyLearningRateScoreDecay)))

(defn batch-size
  "The current inputs batch size"
  [model]
  (.batchSize model))

(defn clear!
  "Clear input"
  [model]
  (doto model
      (.clear)))

(defn compute-gradient-and-score!
  "Update the score"
  [model]
  (doto model
    (.computeGradientAndScore)))

(defn conf
  "The configuration for the neural network"
  [model]
  (.conf model))

(defn fit-model!
  "All models have a fit method

  data is an INDarray of the data you want to fit the model to"
  [& {:keys [model data]
      :as opts}]
  (assert (contains? opts :model) "you must supply a model to fit")
  (cond
    (contains? opts :data)
    (doto model
      (.fit data))
    :else
    (doto model
     (.fit))))

(defn get-optimizer
  "Returns this models optimizer"
  [model]
  (.getOptimizer model))

(defn get-param
  "Get the parameter"
  [& {:keys [model param]}]
  (.getParam model param))

(defn gradient
  "Calculate a gradient"
  [model]
  (.gradient model))

(defn gradient-and-score
  "Get the gradient and score"
  [model]
  (.gradientAndScore model))

(defn init!
  [model]
  (doto model
    (.init)))

(defn input
  "The input/feature matrix for the model"
  [model]
  (.input model))

(defn iterate-once!
  "Run one iteration"
  [& {:keys [model input]}]
  (doto model
    (.iterate input)))

(defn num-params
  "the number of parameters for the model"
  [& {:keys [model backwards?]
      :as opts}]
  (assert (contains? opts :model) "you must supply a model")
  (cond (contains? opts :backwards?)
        (.numParams model backwards?)
        :else
        (.numParams model)))

(defn params
  "Parameters of the model (if any)"
  [model]
  (.params model))

(defn param-table
  "The param table"
  [& {:keys [model backprop-params-only?]
      :as opts}]
  (assert (contains? opts :model) "you must supply a model")
  (cond (contains? opts :backprop-params-only?)
        (.paramTable model backprop-params-only?)
        :else
        (.paramTable model)))

(defn score
  "The score for the model"
  [model]
  (.score model))

(defn set-conf!
  "Setter for the configuration"
  [model]
  (doto model
    (.setConf conf)))

(defn set-listeners!
  "set the iteration listeners for the computational graph"
  [& {:keys [model listeners]}]
  (doto model
    (.setListeners listeners)))

(defn set-param!
  "Set the parameter with a new ndarray"
  [& {:keys [model k v]
      :as opts}]
  (doto model
    (.setParam k v)))

(defn set-params!
  "Set the parameters for this model."
  [& {:keys [model params]}]
  (doto model
    (.setParams params)))

(defn set-param-table!
  "Setter for the param table"
  [& {:keys [model param-table-map]}]
  (doto model (.setParamTable param-table-map)))

(defn update!
  "if gradient comes from deafult-gradient-implementation:
   - Update layer weights and biases with gradient change

  if gradient is an INDArray and param-type is supplied:
   -Perform one update applying the gradient"
  [& {:keys [model gradient param-type]
      :as opts}]
  (assert (contains-many? opts :model :gradient)
          "you must supply a model and a gradient")
  (if (contains? opts :param-type)
    (.update model gradient param-type)
    (.update model gradient)))
