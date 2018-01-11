(ns ^{:doc "A Model is meant for predicting something from data.
 Note that this is not like supervised learning where there are labels attached to the examples.
 see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Model.html"}
  dl4clj.nn.api.model
  (:import [org.deeplearning4j.nn.api Model])
  (:require [dl4clj.utils :refer [contains-many? obj-or-code? eval-if-code]]
            [dl4clj.helpers :refer [reset-if-empty?!]]
            [clojure.core.match :refer [match]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; getters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn batch-size
  "The current inputs batch size"
  [model & {:keys [as-code?]
            :or {as-code? true}}]
  (match [model]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.batchSize ~model))
         :else
         (.batchSize model)))

(defn conf
  "The configuration for the neural network"
  [model & {:keys [as-code?]
            :or {as-code? true}}]
  (match [model]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.conf ~model))
         :else
         (.conf model)))

(defn get-optimizer
  "Returns this models optimizer"
  [model & {:keys [as-code?]
            :or {as-code? true}}]
  (match [model]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getOptimizer ~model))
         :else
         (.getOptimizer model)))

(defn get-param
  "Get the parameter"
  [& {:keys [model param as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :param (:or (_ :guard string?)
                       (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getParam ~model ~param))
         :else
         (let [[p] (eval-if-code [param seq? string?])]
           (.getParam model p))))

(defn gradient-and-score
  "Get the gradient and score"
  [model & {:keys [as-code?]
            :or {as-code? true}}]
  (match [model]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.gradientAndScore ~model))
         :else
         (.gradientAndScore model)))

(defn input
  "returns the input/feature matrix for the model"
  [model & {:keys [as-code?]
            :or {as-code? true}}]
  (match [model]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.input ~model))
         :else
         (.input model)))

(defn num-params
  "the number of parameters for the model
   - 1 x m vector where the vector is composed of a flattened vector of all
     of the weights for the various neuralNets and output layer"
  [& {:keys [model backwards? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :backwards (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.numParams ~model ~backwards?))
         [{:model _
           :backwards _}]
         (let [[b?] (eval-if-code [backwards? seq? boolean?])]
           (.numParams model b?))
         [{:model (_ :guard seq?)}]
         (obj-or-code? as-code? `(.numParams ~model))
         :else
         (.numParams model)))

(defn params
  "Returns a 1 x m vector where the vector is composed of a flattened vector
  of all of the weights for the various neuralNets(w,hbias NOT VBIAS)
  and output layer"
  [& {:keys [model backward-only? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :backward-only (:or (_ :guard boolean?)
                               (_ :guard seq?))}]

         (obj-or-code? as-code? `(.params ~model ~backward-only?))
         [{:model _
           :backward-only _}]
         (let [[b?] (eval-if-code [backward-only? seq? boolean?])]
           (.params model b?))
         [{:model (_ :guard seq?)}]
         (obj-or-code? as-code? `(.params ~model))
         :else
         (.params model)))

(defn param-table
  "The param table"
  [& {:keys [model backprop-params-only? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :backprop-params-only? (:or (_ :guard boolean?)
                                       (_ :guard seq?))}]
         (obj-or-code? as-code? `(.paramTable ~model ~backprop-params-only?))
         [{:model _
           :backprop-params-only? _}]
         (let [[b?] (eval-if-code [backprop-params-only? seq? boolean?])]
           (.paramTable model b?))
         [{:model (_ :guard seq?)}]
         (obj-or-code? as-code? `(.paramTable ~model))
         :else
         (.paramTable model)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; setters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn accumulate-score!
  "Sets a rolling tally for the score."
  [& {:keys [model accum as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :accum (:or (_ :guard number?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~model (.accumulateScore (double ~accum))))
         :else
         (let [[a] (eval-if-code [accum seq? number?])]
           (doto model (.accumulateScore a)))))

(defn set-conf!
  "Setter for the configuration"
  [& {:keys [model conf as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :conf (_ :guard seq?)}]
         (obj-or-code?
          as-code?
          `(doto ~model
            (.setConf ~conf)))
         :else
         (let [[m c] (eval-if-code [model seq?]
                                   [conf seq?])]
          (doto m (.setConf c)))))

(defn set-listeners!
  "set the iteration listeners for the computational graph"
  ;; this may cause issues in the future
  [& {:keys [model listeners as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :listeners (_ :guard vector?)}]
         (obj-or-code?
          as-code?
          `(doto ~model
             (.setListeners ~listeners)))
         [{:model (_ :guard seq?)
           :listeners (_ :guard seq?)}]
         (obj-or-code?
          as-code?
          `(doto ~model
             (.setListeners [~listeners])))
         [{:model _
           :listeners (_ :guard vector?)}]
         (doto model
           (.setListeners listeners))
         :else
         (doto model
           (.setListeners [listeners]))))

(defn set-param!
  "Set the parameter with a new ndarray

  :k (str), the key to set

  :v (INDArray or vec), the value to be set"
  [& {:keys [model k v as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :k (:or (_ :guard string?)
                   (_ :guard seq?))
           :v (:or (_ :guard vector?)
                   (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~model
            (.setParam ~k (vec-or-matrix->indarray ~v))))
         :else
         (let [[m p-name p-value] (eval-if-code [model seq?]
                                                [k seq? string?]
                                                [v seq?])]
          (doto m
           (.setParam p-name (vec-or-matrix->indarray p-value))))))

(defn set-params!
  "Set the parameters for this model."
  [& {:keys [model params as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :params (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~model
            (.setParams (vec-or-matrix->indarray ~params))))
         :else
         (let [[m p] (eval-if-code [model seq?] [params seq?])]
          (doto m
           (.setParams (vec-or-matrix->indarray p))))))

(defn set-param-table!
  "Setter for the param table"
  [& {:keys [model param-table-map as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :param-table-map (:or (_ :guard map?)
                                 (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~model (.setParamTable ~param-table-map)))
         :else
         (let [[m p-table] (eval-if-code [model seq?] [param-table-map seq? map?])]
           (doto m (.setParamTable p-table)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn clear-model!
  "Clear input"
  [model & {:keys [as-code?]
            :or {as-code? true}}]
  (match [model]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~model .clear))
         :else
         (doto model .clear)))

(defn init-params!
  "Initialize the parameters and return the model"
  [model & {:keys [as-code?]
            :or {as-code? true}}]
  (match [model]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(doto ~model .initParams))
         :else
         (doto model .initParams)))

(defn fit!
  "Fit/train the model

  if you supply an iterator, it is only reset if it is at the end of the collection"
  ;; data is used for unsupervised training
  [& {:keys [mln dataset iter data labels
             features features-mask labels-mask
             examples label-idxs as-code?]
      :or {as-code? true}
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
         (obj-or-code?
          as-code?
          `(doto ~mln
           (.fit (vec-or-matrix->indarray ~features)
                 (vec-or-matrix->indarray ~labels)
                 (vec-or-matrix->indarray ~features-mask)
                 (vec-or-matrix->indarray ~labels-mask))))
         [{:mln _ :features _ :labels _ :features-mask _ :labels-mask _}]
         (let [[m f l f-mask l-mask]
               (eval-if-code [mln seq?]
                             [features seq?]
                             [labels seq?]
                             [features-mask seq?]
                             [labels-mask seq?])]
          (doto m
           (.fit (vec-or-matrix->indarray f)
                 (vec-or-matrix->indarray l)
                 (vec-or-matrix->indarray f-mask)
                 (vec-or-matrix->indarray l-mask))))
         [{:mln (_ :guard seq?)
           :examples (:or (_ :guard vector?)
                          (_ :guard seq?))
           :label-idxs (:or (_ :guard vector?)
                            (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~mln
           (.fit (vec-or-matrix->indarray ~examples)
                 (int-array ~label-idxs))))
         [{:mln _ :examples _ :label-idxs _}]
         (let [[m e l-idxs] (eval-if-code [mln seq?]
                                          [examples seq?]
                                          [label-idxs seq? vector?])]
          (doto m (.fit (vec-or-matrix->indarray e) (int-array l-idxs))))
         [{:mln (_ :guard seq?)
           :data (:or (_ :guard vector?)
                      (_ :guard seq?))
           :labels (:or (_ :guard vector?)
                        (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~mln
            (.fit (vec-or-matrix->indarray ~data) (vec-or-matrix->indarray ~labels))))
         [{:mln _ :data _ :labels _}]
         (let [[m d l] (eval-if-code [mln seq?] [data seq?] [labels seq?])]
          (doto m (.fit (vec-or-matrix->indarray d) (vec-or-matrix->indarray l))))
         [{:mln (_ :guard seq?)
           :data (:or (_ :guard vector?)
                      (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~mln
            (.fit (vec-or-matrix->indarray ~data))))
         [{:mln _ :data _}]
         (let [[m d] (eval-if-code [mln seq?] [data seq?])]
          (doto m (.fit (vec-or-matrix->indarray d))))
         [{:mln (_ :guard seq?) :iter (_ :guard seq?)}]
         (obj-or-code?
          as-code?
          `(doto ~mln (.fit ~iter)))
         [{:mln _ :iter _}]
         (let [[m i] (eval-if-code [mln seq?] [iter seq?])]
          (doto m
           (.fit (reset-if-empty?! i))))
         [{:mln (_ :guard seq?) :dataset (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~mln (.fit ~dataset)))
         [{:mln _ :dataset _}]
         (let [[m d] (eval-if-code [mln seq?] [dataset seq?])]
           (doto m (.fit d)))
         [{:mln (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~mln .fit))
         :else
         (doto mln .fit)))

(defn calc-gradient
  "Calculate a gradient"
  [model & {:keys [as-code?]
            :or {as-code? true}}]
  (match [model]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.gradient ~model))
         :else
         (.gradient model)))

(defn init!
  "initialize the model"
  [& {:keys [model params clone-param-array? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :params (:or (_ :guard seq?)
                        (_ :guard vector?))
           :clone-param-array? (:or (_ :guard boolean?)
                                    (_ :guard seq?))}]
         (obj-or-code? as-code? `(doto ~model (.init (vec-or-matrix->indarray ~params)
                                                     ~clone-param-array?)))
         [{:model _ :params _ :clone-param-array? _}]
         (let [[m p clone?] (eval-if-code [model seq?] [params seq?]
                                          [clone-param-array? seq? boolean?])]
          (doto m (.init (vec-or-matrix->indarray p) clone?)))
         [{:model (_ :guard seq?)}]
         (obj-or-code? as-code? `(doto ~model .init))
         :else
         (doto model .init)))

(defn iterate-once!
  "Run one iteration"
  [& {:keys [model input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~model (.iterate (vec-or-matrix->indarray ~input))))
         :else
         (let [[m i] (eval-if-code [model seq?] [input seq?])]
           (doto m (.iterate (vec-or-matrix->indarray i))))))

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
  [& {:keys [model dataset training? return-model? as-code?]
      :or {return-model? false
           as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))
           :return-model? true}]
         (obj-or-code? as-code? `(doto ~model (.score ~dataset ~training?)))
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))
           :return-model? false}]
         (obj-or-code? as-code? `(.score ~model ~dataset ~training?))
         [{:model _ :dataset _ :training? _ :return-model? true}]
         (let [[m d t?] (eval-if-code [model seq?] [dataset seq?]
                                      [training? seq? boolean?])]
           (doto m (.score d t?)))
         [{:model _ :dataset _ :training? _ :return-model? false}]
         (let [[m d t?] (eval-if-code [model seq?] [dataset seq?]
                                      [training? seq? boolean?])]
           (.score m d t?))
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.score ~model ~dataset ~training?))
         [{:model _ :dataset _ :training? _}]
         (let [[m d t?] (eval-if-code [model seq?] [dataset seq?]
                                      [training? seq? boolean?])]
           (.score m d t?))
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)
           :return-model? true}]
         (obj-or-code? as-code? `(doto ~model (.score ~dataset)))
         [{:model _ :dataset _ :return-model? true}]
         (let [[m d] (eval-if-code [model seq?] [dataset seq?])]
           (doto m (.score d)))
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)
           :return-model? false}]
         (obj-or-code? as-code? `(.score ~model ~dataset))
         [{:model _ :dataset _ :return-model? false}]
         (let [[m d] (eval-if-code [model seq?] [dataset seq?])]
           (.score m d))
         [{:model (_ :guard seq?)
           :dataset (_ :guard seq?)}]
         (obj-or-code? as-code? `(.score ~model ~dataset))
         [{:model _ :dataset _}]
         (let [[m d] (eval-if-code [model seq?] [dataset seq?])]
           (.score m d))
         [{:model (_ :guard seq?) :return-model? true}]
         (obj-or-code? as-code? `(doto ~model .score))
         [{:model _ :return-model? true}]
         (doto model .score)
         :else
         (.score model)))

(defn update!
  "if gradient comes from deafult-gradient-implementation:
   - Update layer weights and biases with gradient change

  if gradient is an INDArray or vec and param-type is supplied:
   -Perform one update applying the gradient"
  [& {:keys [model gradient param-type as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :gradient (:or (_ :guard vector?)
                          (_ :guard seq?))
           :param-type (:or (_ :guard string?)
                            (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.update ~model (vec-or-matrix->indarray ~gradient) ~param-type))
         [{:model _
           :gradient _
           :param-type _}]
         (let [[m g p-type] (eval-if-code [model seq?] [gradient seq?]
                                          [param-type seq? string?])]
           (.update m (vec-or-matrix->indarray g) p-type))
         [{:model (_ :guard seq?)
           :gradient (:or (_ :guard vector?)
                          (_ :guard seq?))}]
         (obj-or-code? as-code? `(.update ~model (vec-or-matrix->indarray ~gradient)))
         :else
         (let [[m g] (eval-if-code [model seq?] [gradient seq?])]
           (.update m (vec-or-matrix->indarray g)))))
