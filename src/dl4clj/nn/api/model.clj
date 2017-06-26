(ns ^{:doc "A Model is meant for predicting something from data.
 Note that this is not like supervised learning where there are labels attached to the examples.
 see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Model.html"}
  dl4clj.nn.api.model
  (:import [org.deeplearning4j.nn.api Model])
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.helpers :refer [reset-if-empty?!]]))

;; add .setBackpropGradientsViewArray and .validateInput

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

(defn init-params!
  "Initialize the parameters and return the model"
  [model]
  (doto model (.initParams)))

(defn conf
  "The configuration for the neural network"
  [model]
  (.conf model))

(defn fit!
  "Fit/train the model"
  [& {:keys [mln ds iter data labels
             features features-mask labels-mask
             examples label-idxs]
      :as opts}]
  (cond (contains-many? opts :features :labels :features-mask :labels-mask)
        (doto mln
          (.fit features labels features-mask labels-mask))
        (contains-many? opts :examples :label-idxs)
        (doto mln
          (.fit examples label-idxs))
        (contains-many? opts :data :labels)
        (doto mln
          (.fit data labels))
        (contains? opts :data)
        (doto mln
          (.fit data))
        (contains? opts :iter)
        (doto mln
          (.fit (reset-if-empty?! iter)))
        (contains? opts :ds)
        (doto mln (.fit ds))
        :else
        (doto mln (.fit))))

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
  "initialize the model"
  [& {:keys [model params clone-param-array?]
      :as opts}]
  (if (contains-many? opts :params :clone-param-array?)
    (doto model (.init params clone-param-array?))
    (doto model (.init))))

(defn input
  "returns the input/feature matrix for the model"
  [model]
  (.input model))

(defn iterate-once!
  "Run one iteration"
  [& {:keys [model input]}]
  (doto model
    (.iterate input)))

(defn num-params
  "the number of parameters for the model
   - 1 x m vector where the vector is composed of a flattened vector of all
     of the weights for the various neuralNets and output layer"
  [& {:keys [model backwards?]
      :as opts}]
  (assert (contains? opts :model) "you must supply a model")
  (cond (contains? opts :backwards?)
        (.numParams model backwards?)
        :else
        (.numParams model)))

(defn params
  "Returns a 1 x m vector where the vector is composed of a flattened vector
  of all of the weights for the various neuralNets(w,hbias NOT VBIAS)
  and output layer"
  [& {:keys [model backward-only?]
      :as opts}]
  (if (contains? opts :backward-only?)
    (.params model backward-only?)
    (.params model)))

(defn param-table
  "The param table"
  [& {:keys [model backprop-params-only?]
      :as opts}]
  (assert (contains? opts :model) "you must supply a model")
  (cond (contains? opts :backprop-params-only?)
        (.paramTable model backprop-params-only?)
        :else
        (.paramTable model)))

(defn score!
  "only model supplied: Score of the model (relative to the objective function)

   model and dataset supplied: Sets the input and labels and returns a score for
   the prediction with respect to the true labels

   model, dataset and training? supplied: Calculate the score (loss function)
   of the prediction with respect to the true labels

   :dataset (ds), a dataset
   -see: nd4clj.linalg.dataset.data-set
         dl4clj.datasets.datavec

   :training? (boolean), are we in training mode?

   :return-model? (boolean), if you want to return the scored model instead of the score"
  [& {:keys [model dataset training? return-model?]
      :or {return-model? false}
      :as opts}]
  (cond (and (contains-many? opts :dataset :training?)
             (true? return-model?))
        (doto model (.score dataset training?))
        (contains-many? opts :dataset :training?)
        (.score model dataset training?)
        (and (contains? opts :dataset)
             (true? return-model?))
        (doto model (.score dataset))
        (contains? opts :dataset)
        (.score model dataset)
        (true? return-model?)
        (doto model (.score))
        :else
        (.score model)))

(defn set-conf!
  "Setter for the configuration"
  [& {:keys [model conf]}]
  (doto model
    (.setConf conf)))

(defn set-listeners!
  "set the iteration listeners for the computational graph"
  [& {:keys [model listeners]}]
  (doto model
    (.setListeners listeners)))

(defn set-param!
  "Set the parameter with a new ndarray

  :k (str), the key to set

  :v (INDArray), the value to be set"
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
