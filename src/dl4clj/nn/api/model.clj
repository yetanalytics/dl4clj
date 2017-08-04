(ns ^{:doc "A Model is meant for predicting something from data.
 Note that this is not like supervised learning where there are labels attached to the examples.
 see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Model.html"}
  dl4clj.nn.api.model
  (:import [org.deeplearning4j.nn.api Model])
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.helpers :refer [reset-if-empty?!]]
            [clojure.core.match :refer [match]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

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

(defn clear-model!
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
  "Fit/train the model

  if you supply an iterator, it is only reset if it is at the end of the collection"
  [& {:keys [mln dataset iter data labels
             features features-mask labels-mask
             examples label-idxs]
      :as opts}]
  (let [d (vec-or-matrix->indarray data)]
    (match [opts]
           [{:mln _ :features _ :labels _ :features-mask _ :labels-mask _}]
           (doto mln
             (.fit (vec-or-matrix->indarray features)
                   (vec-or-matrix->indarray labels)
                   (vec-or-matrix->indarray features-mask)
                   (vec-or-matrix->indarray labels-mask)))
           [{:mln _ :examples _ :label-idxs _}]
           (doto mln
             (.fit (vec-or-matrix->indarray examples)
                   (int-array label-idxs)))
           [{:mln _ :data _ :labels _}]
           (doto mln
             (.fit d (vec-or-matrix->indarray labels)))
           ;; this might be the same thing as just examples?
           [{:mln _ :data _}]
           (doto mln
             (.fit d))
           [{:mln _ :iter _}]
           (doto mln
             (.fit (reset-if-empty?! iter)))
           [{:mln _ :dataset _}]
           (doto mln (.fit dataset))
           :else
           (doto mln .fit))))

(defn get-optimizer
  "Returns this models optimizer"
  [model]
  (.getOptimizer model))

(defn get-param
  "Get the parameter"
  [& {:keys [model param]}]
  (.getParam model param))

(defn calc-gradient
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
  (match [opts]
         [{:model _ :params _ :clone-param-array? _}]
         (doto model (.init params clone-param-array?))
         :else
         (doto model .init)))

(defn input
  "returns the input/feature matrix for the model"
  [model]
  (.input model))

(defn iterate-once!
  "Run one iteration"
  [& {:keys [model input]}]
  (doto model
    (.iterate (vec-or-matrix->indarray input))))

(defn num-params
  "the number of parameters for the model
   - 1 x m vector where the vector is composed of a flattened vector of all
     of the weights for the various neuralNets and output layer"
  [& {:keys [model backwards?]
      :as opts}]
  (if (boolean? backwards?)
    (.numParams model backwards?)
    (.numParams model)))

(defn params
  "Returns a 1 x m vector where the vector is composed of a flattened vector
  of all of the weights for the various neuralNets(w,hbias NOT VBIAS)
  and output layer"
  [& {:keys [model backward-only?]
      :as opts}]
  (if (boolean? backward-only?)
    (.params model backward-only?)
    (.params model)))

(defn param-table
  "The param table"
  [& {:keys [model backprop-params-only?]
      :as opts}]
  (if (boolean? backprop-params-only?)
        (.paramTable model backprop-params-only?)
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
  (match [opts]
         [{:model _ :dataset _ :training? _ :return-model? true}]
         (doto model (.score dataset training?))
         [{:model _ :dataset _ :training? _ :return-model? false}]
         (.score model dataset training?)
         [{:model _ :dataset _ :training? _}]
         (.score model dataset training?)
         [{:model _ :dataset _ :return-model? true}]
         (doto model (.score dataset))
         [{:model _ :dataset _ :return-model? false}]
         (.score model dataset)
         [{:model _ :dataset _}]
         (.score model dataset)
         [{:model _ :return-model? true}]
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

  :v (INDArray or vec), the value to be set"
  [& {:keys [model k v]
      :as opts}]
  (doto model
    (.setParam k (vec-or-matrix->indarray v))))

(defn set-params!
  "Set the parameters for this model."
  [& {:keys [model params]}]
  (doto model
    (.setParams (vec-or-matrix->indarray params))))

(defn set-param-table!
  "Setter for the param table"
  [& {:keys [model param-table-map]}]
  (doto model (.setParamTable param-table-map)))

(defn update!
  "if gradient comes from deafult-gradient-implementation:
   - Update layer weights and biases with gradient change

  if gradient is an INDArray or vec and param-type is supplied:
   -Perform one update applying the gradient"
  [& {:keys [model gradient param-type]
      :as opts}]
  (let [g (vec-or-matrix->indarray gradient)]
    (if param-type
      (.update model g param-type)
      (.update model g))))
