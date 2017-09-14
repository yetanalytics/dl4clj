(ns ^{:doc "A Model is meant for predicting something from data.
 Note that this is not like supervised learning where there are labels attached to the examples.
 see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Model.html"}
  dl4clj.nn.api.model
  (:import [org.deeplearning4j.nn.api Model])
  (:require [dl4clj.utils :refer [contains-many?]]
            [dl4clj.helpers :refer [reset-if-empty?!]]
            [clojure.core.match :refer [match]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; getters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn batch-size
  "The current inputs batch size"
  [model]
  (match [model]
         [(_ :guard seq?)]
         `(.batchSize ~model)
         :else
         (.batchSize model)))

(defn conf
  "The configuration for the neural network"
  [model]
  (match [model]
         [(_ :guard seq?)]
         `(.conf ~model)
         :else
         (.conf model)))

(defn get-optimizer
  "Returns this models optimizer"
  [model]
  (match [model]
         [(_ :guard seq?)]
         `(.getOptimizer ~model)
         :else
         (.getOptimizer model)))

(defn get-param
  "Get the parameter"
  [& {:keys [model param]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :param (:or (_ :guard string?)
                       (_ :guard seq?))}]
         `(.getParam ~model ~param)
         :else
         (.getParam model param)))

(defn gradient-and-score
  "Get the gradient and score"
  [model]
  (match [model]
         [(_ :guard seq?)]
         `(.gradientAndScore ~model)
         :else
         (.gradientAndScore model)))

(defn input
  "returns the input/feature matrix for the model"
  [model]
  (match [model]
         [(_ :guard seq?)]
         `(.input ~model)
         :else
         (.input model)))

(defn num-params
  "the number of parameters for the model
   - 1 x m vector where the vector is composed of a flattened vector of all
     of the weights for the various neuralNets and output layer"
  [& {:keys [model backwards?]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :backwards (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         `(.numParams ~model ~backwards?)
         [{:model _
           :backwards _}]
         (.numParams model backwards?)
         [{:model (_ :guard seq?)}]
         `(.numParams ~model)
         :else
         (.numParams model)))

(defn params
  "Returns a 1 x m vector where the vector is composed of a flattened vector
  of all of the weights for the various neuralNets(w,hbias NOT VBIAS)
  and output layer"
  [& {:keys [model backward-only?]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :backward-only (:or (_ :guard boolean?)
                               (_ :guard seq?))}]

         `(.params ~model ~backward-only?)
         [{:model _
           :backward-only _}]
         (.params model backward-only?)
         [{:model (_ :guard seq?)}]
         `(.params ~model)
         :else
         (.params model)))

(defn param-table
  "The param table"
  [& {:keys [model backprop-params-only?]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :backprop-params-only? (:or (_ :guard boolean?)
                                       (_ :guard seq?))}]
         `(.paramTable ~model ~backprop-params-only?)
         [{:model _
           :backprop-params-only? _}]
         (.paramTable model backprop-params-only?)
         [{:model (_ :guard seq?)}]
         `(.paramTable ~model)
         :else
         (.paramTable model)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; setters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn accumulate-score!
  "Sets a rolling tally for the score."
  [& {:keys [model accum]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :accum (:or (_ :guard number?)
                       (_ :guard seq?))}]
         `(doto ~model
            (.accumulateScore (double ~accum)))
         :else
         (doto model
           (.accumulateScore accum))))

(defn set-conf!
  "Setter for the configuration"
  [& {:keys [model conf]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :conf (_ :guard seq?)}]
         `(doto ~model
            (.setConf ~conf))
         :else
         (doto model
           (.setConf conf))))

(defn set-listeners!
  "set the iteration listeners for the computational graph"
  [& {:keys [model listeners]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :listeners _}]
         `(doto ~model
            (.setListeners ~listeners))
         :else
         (doto model
           (.setListeners listeners))))

(defn set-param!
  "Set the parameter with a new ndarray

  :k (str), the key to set

  :v (INDArray or vec), the value to be set"
  [& {:keys [model k v]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :k (:or (_ :guard string?)
                   (_ :guard seq?))
           :v (:or (_ :guard vector?)
                   (_ :guard seq?))}]
         `(doto ~model
            (.setParam ~k (vec-or-matrix->indarray ~v)))
         :else
         (doto model
           (.setParam k (vec-or-matrix->indarray v)))))

(defn set-params!
  "Set the parameters for this model."
  [& {:keys [model params]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :params (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         `(doto ~model
            (.setParams (vec-or-matrix->indarray ~params)))
         :else
         (doto model
           (.setParams (vec-or-matrix->indarray params)))))

(defn set-param-table!
  "Setter for the param table"
  [& {:keys [model param-table-map]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :param-table-map (:or (_ :guard map?)
                                 (_ :guard seq?))}]
         `(doto ~model (.setParamTable ~param-table-map))
         :else
         (doto model (.setParamTable param-table-map))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn clear-model!
  "Clear input"
  [model]
  (match [model]
         [(_ :guard seq?)]
         `(doto ~model
            .clear)
         :else
         (doto model
           .clear)))

(defn init-params!
  "Initialize the parameters and return the model"
  [model]
  (match [model]
         [(_ :guard seq?)]
         `(doto ~model .initParams)
         :else
         (doto model .initParams)))

(defn fit!
  "Fit/train the model

  if you supply an iterator, it is only reset if it is at the end of the collection"
  ;; data is used for unsupervised training
  [& {:keys [mln dataset iter data labels
             features features-mask labels-mask
             examples label-idxs]
      :as opts}]
  (match [opts]
         [{:mln (_ :guard seq?)
           :features (:or (_ :guard vector?)
                          (_ :guard seq?))
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))
           :features-mask (:or (_ :guard vector?)
                               (_ :guard seq?))
           :labels-mask (:or (_ :guard vector?)
                             (_ :guard seq?))}]
         `(doto ~mln
           (.fit (vec-or-matrix->indarray ~features)
                 (vec-or-matrix->indarray ~labels)
                 (vec-or-matrix->indarray ~features-mask)
                 (vec-or-matrix->indarray ~labels-mask)))
         [{:mln _ :features _ :labels _ :features-mask _ :labels-mask _}]
         (doto mln
           (.fit (vec-or-matrix->indarray features)
                 (vec-or-matrix->indarray labels)
                 (vec-or-matrix->indarray features-mask)
                 (vec-or-matrix->indarray labels-mask)))
         [{:mln (_ :guard seq?)
           :examples (:or (_ :guard vector?)
                          (_ :guard seq?))
           :label-idxs (:or (_ :guard vector?)
                            (_ :guard seq?))}]
         `(doto ~mln
           (.fit (vec-or-matrix->indarray ~examples)
                 (int-array ~label-idxs)))
         [{:mln _ :examples _ :label-idxs _}]
         (doto mln
           (.fit (vec-or-matrix->indarray examples)
                 (int-array label-idxs)))
         [{:mln (_ :guard seq?)
           :data (:or (_ :guard vector?)
                      (_ :guard seq?))
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         `(doto ~mln
            (.fit (vec-or-matrix->indarray ~data) (vec-or-matrix->indarray ~labels)))
         [{:mln _ :data _ :labels _}]
         (doto mln
           (.fit (vec-or-matrix->indarray data) (vec-or-matrix->indarray labels)))
         [{:mln (_ :guard seq?)
           :data (:or (_ :guard vector?)
                      (_ :guard seq?))}]
         `(doto ~mln
            (.fit (vec-or-matrix->indarray ~data)))
         [{:mln _ :data _}]
         (doto mln
           (.fit (vec-or-matrix->indarray data)))
         [{:mln (_ :guard seq?) :iter (_ :guard seq?)}]
         `(doto ~mln
            (.fit ~iter))
         [{:mln _ :iter _}]
         (doto mln
           (.fit (reset-if-empty?! iter)))
         [{:mln (_ :guard seq?) :dataset (_ :guard seq?)}]
         `(doto ~mln (.fit ~dataset))
         [{:mln _ :dataset _}]
         (doto mln (.fit dataset))
         [{:mln (_ :guard seq?)}]
         `(doto ~mln .fit)
         :else
         (doto mln .fit)))

(defn calc-gradient
  "Calculate a gradient"
  [model]
  (match [model]
         [(_ :guard seq?)]
         `(.gradient ~model)
         :else
         (.gradient model)))

(defn init!
  "initialize the model"
  [& {:keys [model params clone-param-array?]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :params _ ;; need to double check but belive it should be vec-or-matrix->indarray
           :clone-param-array? (:or (_ :guard boolean?)
                                    (_ :guard seq?))}]
         `(doto ~model (.init ~params ~clone-param-array?))
         [{:model _ :params _ :clone-param-array? _}]
         (doto model (.init params clone-param-array?))
         [{:model (_ :guard seq?)}]
         `(doto ~model .init)
         :else
         (doto model .init)))

(defn iterate-once!
  "Run one iteration"
  [& {:keys [model input]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         `(doto ~model
            (.iterate (vec-or-matrix->indarray ~input)))
         :else
         (doto model
           (.iterate (vec-or-matrix->indarray input)))))

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
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))
           :return-model? true}]
         `(doto ~model (.score ~dataset ~training?))
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))
           :return-model? false}]
         `(.score ~model ~dataset ~training?)
         [{:model _ :dataset _ :training? _ :return-model? true}]
         (doto model (.score dataset training?))
         [{:model _ :dataset _ :training? _ :return-model? false}]
         (.score model dataset training?)
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         `(.score ~model ~dataset ~training?)
         [{:model _ :dataset _ :training? _}]
         (.score model dataset training?)
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)
           :return-model? true}]
         `(doto ~model (.score ~dataset))
         [{:model _ :dataset _ :return-model? true}]
         (doto model (.score dataset))
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)
           :return-model? false}]
         `(.score ~model ~dataset)
         [{:model _ :dataset _ :return-model? false}]
         (.score model dataset)
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)}]
         `(.score ~model ~dataset)
         [{:model _ :dataset _}]
         (.score model dataset)
         [{:model (_ :guard seq?) :return-model? true}]
         `(doto ~model .score)
         [{:model _ :return-model? true}]
         (doto model .score)
         :else
         (.score model)))

(defn update!
  "if gradient comes from deafult-gradient-implementation:
   - Update layer weights and biases with gradient change

  if gradient is an INDArray or vec and param-type is supplied:
   -Perform one update applying the gradient"
  [& {:keys [model gradient param-type]
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :gradient (:or (_ :guard vector?)
                          (_ :guard seq?))
           :param-type (:or (_ :guard string?)
                            (_ :guard seq?))}]
         `(.update ~model (vec-or-matrix->indarray ~gradient) ~param-type)
         [{:model _
           :gradient _
           :param-type _}]
         (.update model (vec-or-matrix->indarray gradient) param-type)
         [{:model (_ :guard seq?)
           :gradient (:or (_ :guard vector?)
                          (_ :guard seq?))}]
         `(.update ~model (vec-or-matrix->indarray ~gradient))
         :else
         (.update model (vec-or-matrix->indarray gradient))))
