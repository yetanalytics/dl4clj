(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Model.html"}
  dl4clj.nn.api.model
  (:import [org.deeplearning4j.nn.api Model]))

(defn accumulate-score
  "Sets a rolling tally for the score."
  [^Model this accum]
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
  ([^Model this]
   (.fit this))
  ([^Model this data]
   (.fit this data)))

(defn get-optimizer
  "Returns this models optimizer"
  [^Model this]
  (.getOptimizer this))

(defn get-param
  "Get the parameter"
  [^Model this param]
  (.getParam this (name param)))

(defn gradient
  "Calculate a gradient"
  [^Model this param]
  (.gradient this))

(defn gradient-and-score
  "Get the gradient and score"
  [^Model this]
  (.gradientAndScore this))


(defn init-params
  "Initialize the parameters"
  [^Model this]
  (.initParams this))

(defn input
  "The input/feature matrix for the model"
  [^Model this]
  (.input this))

(defn iterate
  "Run one iteration"
  [^Model this input]
  (.iterate this input))

(defn num-params
  "the number of parameters for the model"
  ([^Model this]
   (.numParams this))
  ([^Model this backwards?]
   (.numParams this (boolean backwards?))))

(defn params
  "Parameters of the model (if any)"
  [^Model this]
  (.params this))

(defn param-table
  "The param table"
  [^Model this]
  (.paramTable this))

(defn score
  "The score for the model"
  [^Model this]
  (.score this))

(defn set-conf
  "Setter for the configuration"
  [^Model this]
  (.setConf this conf))

(defn set-param
  "Set the parameter with a new ndarray"
  [^Model this key val]
  (.setParam this (name key) val))

(defn setParams
  "Set the parameters for this model."
  [^Model this params]
  (.setParams this params))

(defn set-param-table
  "Setter for the param table"
  [^Model this param-table]
  (.setParamTable this param-table))

(defn update
  "Perform one update applying the gradient"
  [^Model this gradient param-type]
  (.update this gradient (name param-type)))

(defn validateInput
  "Validate the input"
  [^Model this]
  (.validateInput this))

