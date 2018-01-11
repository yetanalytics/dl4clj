(ns ^{:doc "fns from the dl4j Interface for a layer of a neural network.
 This has an activation function, an input and output size, weights, and a bias
 see http://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Layer.html"}
  dl4clj.nn.api.layer
  (:import [org.deeplearning4j.nn.api Layer])
  (:require [dl4clj.utils :refer [contains-many? obj-or-code? eval-if-code]]
            [dl4clj.constants :as enum]
            [clojure.core.match :refer [match]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; getters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-index
  "Get the layer index."
  [layer & {:keys [as-code?]
            :or {as-code? true}}]
  (match [layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getIndex ~layer))
         :else
         (.getIndex layer)))

(defn get-input-mini-batch-size
  "Get current/last input mini-batch size, as set by set-input-mini-batch-size!"
  [layer & {:keys [as-code?]
            :or {as-code? true}}]
  (match [layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getInputMiniBatchSize ~layer))
         :else
         (.getInputMiniBatchSize layer)))

(defn get-listeners
  "Get the iteration listeners for this layer."
  [layer & {:keys [as-code?]
            :or {as-code? true}}]
  (match [layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getListeners ~layer))
         :else
         (.getListeners layer)))

(defn get-mask-array
  "get the mask array"
  [layer & {:keys [as-code?]
            :or {as-code? true}}]
  (match [layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.getMaskArray ~layer))
         :else
         (.getMaskArray layer)))

(defn get-layer-type
  "Returns the layer type"
  [layer & {:keys [as-code?]
            :or {as-code? true}}]
  (match [layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.type ~layer))
         :else
         (.type layer)))

(defn get-l1-by-param-layer
  "Get the L1 coefficient for the given parameter."
  [& {:keys [layer param-name as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :param-name (:or (_ :guard string?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getL1ByParam ~layer ~param-name))
         :else
         (let [[p-name] (eval-if-code [param-name seq? string?])]
           (.getL1ByParam layer p-name))))

(defn get-l2-by-param-layer
  "Get the L2 coefficient for the given parameter."
  [& {:keys [layer param-name as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :param-name (:or (_ :guard string?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getL2ByParam ~layer ~param-name))
         :else
         (let [[p-name] (eval-if-code [param-name seq? string?])]
           (.getL2ByParam layer p-name))))

(defn get-learning-rate-by-param-layer
  "Get the (initial) learning rate coefficient for the given parameter."
  [& {:keys [layer param-name as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :param-name (:or (_ :guard string?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getLearningRateByParam ~layer ~param-name))
         :else
         (let [[p-name] (eval-if-code [param-name seq? string?])]
           (.getLearningRateByParam layer p-name))))

(defn get-output-type
  "returns the output type of the provided layer given the layer input type

  :layer-idx (int), the index of the layer within its model

  :input-type (map), the input to the cnn layer
   - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants"
  [& {:keys [layer layer-idx input-type as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))
           :input-type (:or (_ :guard map?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getOutputType ~layer (int ~layer-idx) (enum/input-types ~input-type)))
         :else
         (let [[l-idx i-type] (eval-if-code [layer-idx seq? number?]
                                            [input-type seq? map?])]
           (.getOutputType layer l-idx (enum/input-types i-type)))))

(defn get-pre-processor-for-input-type
  "For the given type of input to this layer, what preprocessor (if any) is required?

  :input-type (map), the input to the cnn layer
   - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants"
  [& {:keys [layer input-type as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :input-type (:or (_ :guard map?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getPreProcessorForInputType ~layer (enum/input-types ~input-type)))
         :else
         (let [[i-type] (eval-if-code [input-type seq? map?])]
           (.getPreProcessorForInputType layer (enum/input-types i-type)))))

(defn get-updater-by-param
  "Get the updater for the given parameter."
  [& {:keys [layer param-name as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :param-name (:or (_ :guard string?)
                            (_ :guard seq?))}]
         (obj-or-code? as-code? `(.getUpdaterByParam ~layer ~param-name))
         :else
         (let [[p-name] (eval-if-code [param-name seq? string?])]
           (.getUpdaterByParam layer p-name))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; setters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn set-index!
  "Set the layer index and returns the layer.
  :index should be an integer"
  [& {:keys [layer index as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :index (:or (_ :guard number?)
                       (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~layer
            (.setIndex (int ~index))))
         :else
         (let [[i] (eval-if-code [index seq? number?])]
          (doto layer
           (.setIndex (int i))))))

(defn set-layer-input!
  "set the layer's input and return the layer"
  [& {:keys [layer input as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :input (:or (_ :guard seq?)
                       (_ :guard vector?))}]
         (obj-or-code?
          as-code?
          `(doto ~layer
             (.setInput (vec-or-matrix->indarray ~input))))
         :else
         (let [[i l] (eval-if-code [input seq?] [layer seq?])]
          (doto l (.setInput (vec-or-matrix->indarray i))))))

(defn set-input-mini-batch-size!
  "Set current/last input mini-batch size and return the layer.
  Used for score and gradient calculations."
  [& {:keys [layer size as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :size (:or (_ :guard number?)
                      (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~layer
            (.setInputMiniBatchSize (int ~size))))
         :else
         (let [[s] (eval-if-code [size seq? number?])]
          (doto layer
           (.setInputMiniBatchSize s)))))

(defn set-layer-listeners!
  "Set the iteration listeners for the supplied layer and returns the layer.
  listeners is a collection or array of iteration listeners"
  [& {:keys [layer listeners as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :listeners (_ :guard seq?)}]
         (obj-or-code?
          as-code?
          `(doto ~layer
            (.setListeners ~listeners)))
         :else
         (let [[l listenerz] (eval-if-code [layer seq?] [listeners seq?])]
          (doto l
           (.setListeners listenerz)))))

(defn set-mask-array!
  "Set the mask array for this layer and return the layer
  mask-array is an INDArray"
  [& {:keys [layer mask-array as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :mask-array (:or (_ :guard vector?)
                            (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~layer
            (.setMaskArray (vec-or-matrix->indarray ~mask-array))))
         :else
         (let [[m-vec l] (eval-if-code [mask-array seq?] [layer seq?])]
           (doto l (.setMaskArray (vec-or-matrix->indarray m-vec))))))

(defn set-n-in!
  "Set the nIn value (number of inputs, or input depth for CNNs) based on the given input type

  :input-type (map), the input to the cnn layer
   - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants

  :override? (boolean), do you want to override the current n-in?

  returns the layer"
  [& {:keys [layer input-type override? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :input-type (:or (_ :guard map?)
                            (_ :guard seq?))
           :override? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(doto ~layer (.setNIn (enum/input-types ~input-type) ~override?)))
         :else
         (let [[i-type o?] (eval-if-code [input-type seq? map?]
                                         [override? seq? boolean?])]
           (doto layer (.setNIn (enum/input-types i-type) o?)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn initializer
  "returns a param initializer for this layer"
  [layer & {:keys [as-code?]
            :or {as-code? true}}]
  (match [layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.initializer ~layer))
         :else
         (.initializer layer)))

(defn instantiate
  "returns the instantiated layer

  :layer (layer), the layer set in the nn-conf

  :conf (nn-conf), a single layer neural network configuration

  :listener (coll), a collection of listeners for the layer

  :layer-idx (int), the index of the layer within the model

  :layer-param-view (INDArray or vec), the params of the layer

  :initialize-params? (boolean), do you want to initialize the params?"
  [& {:keys [layer conf listener layer-idx as-code?
             layer-param-view initialize-params?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :conf (_ :guard seq?)
           :listener (_ :guard seq?)
           :layer-idx (:or (_ :guard number?)
                           (_ :guard seq?))
           :layer-param-view (:or (_ :guard vector?)
                                  (_ :guard seq?))
           :initialize-params? (:or (_ :guard boolean?)
                                    (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.instantiate ~layer ~conf ~listener (int ~layer-idx)
                        (vec-or-matrix->indarray ~layer-param-view)
                        ~initialize-params?))
         :else
         (let [[l c listen l-idx p-view-vec init-params?]
               (eval-if-code [layer seq?] [conf seq?] [listener seq?]
                             [layer-idx seq? number?] [layer-param-view seq?]
                             [initialize-params? seq? boolean?])]
           (.instantiate l c listen l-idx
                         (vec-or-matrix->indarray p-view-vec)
                         init-params?))))

(defn activate-layer
  "6 opts for triggering an activation
  1) only supply the layer, Trigger an activation with the last specified input

  2) supply training? (boolean), trigger with the last specified input

  3) supply input (INDArray or vec), initialize the layer with the given input and return
   the activation for this layer given this input

  4) supply training-mode (keyword), Trigger an activation with the last specified input
    - keyword is one of :test or :train

  5) supply input and training?

  6) supply input and training-mode
   -both 4 and 5 initialize the layer with the given input and return the activation for this layer given this input"
  [& {:keys [model training? input training-mode as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:model (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.activate ~model (vec-or-matrix->indarray ~input) ~training?))
         [{:model _ :input _ :training? _}]
         (let [[i t? m] (eval-if-code [input seq?] [training? seq? boolean?]
                                      [model seq?])]
          (.activate m (vec-or-matrix->indarray i) t?))
         [{:model (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :training-mode (:or (_ :guard keyword?)
                               (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.activate ~model
                     (vec-or-matrix->indarray ~input)
                     (enum/value-of {:layer-training-mode ~training-mode})))
         [{:model _ :input _ :training-mode _}]
         (let [[i t-m m] (eval-if-code [input seq?] [training-mode seq? keyword?]
                                       [model seq?])]
          (.activate m (vec-or-matrix->indarray i) (enum/value-of {:layer-training-mode t-m})))
         [{:model (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code? as-code? `(.activate ~model (vec-or-matrix->indarray ~input)))
         [{:model _ :input _}]
         (let [[i m] (eval-if-code [input seq?] [model seq?])]
           (.activate m (vec-or-matrix->indarray i)))
         [{:model (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.activate ~model ~training?))
         [{:model _ :training? _}]
         (let [[t?] (eval-if-code [training? seq? boolean?])]
           (.activate model t?))
         [{:model (_ :guard seq?)
           :training-mode (:or (_ :guard keyword?)
                               (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.activate ~model (enum/value-of {:layer-training-mode ~training-mode})))
         [{:model _ :training-mode _}]
         (let [[t-m] (eval-if-code [training-mode seq? keyword?])]
          (.activate model (enum/value-of {:layer-training-mode t-m})))
         [{:model (_ :guard seq?)}]
         (obj-or-code? as-code? `(.activate ~model))
         :else
         (.activate model)))

(defn layer-output
  "returns the output of a layer.

  :input (INDArray or vec), the input to the layer

  :training? (boolean), are we in trianing mode or testing?

  multiple layers implement this fn"
  [& {:keys [layer input training? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.output ~layer (vec-or-matrix->indarray ~input) ~training?))
         [{:layer _ :input _ :training? _}]
         (let [[i t? l] (eval-if-code [input seq?] [training? seq? boolean?]
                                      [layer seq?])]
           (.output l (vec-or-matrix->indarray i) t?))
         [{:layer (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))}]
         (obj-or-code? as-code? `(.output ~layer (vec-or-matrix->indarray ~input)))
         [{:layer _ :input _ }]
         (let [[i l] (eval-if-code [input seq?] [layer seq?])]
           (.output l (vec-or-matrix->indarray i)))
         [{:layer (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.output ~layer ~training?))
         [{:layer _ :training? _}]
         (let [[t?] (eval-if-code [training? seq? boolean?])]
           (.output layer t?))))

(defn feed-forward-mask-array
  "Feed forward the input mask array, setting in in the layer as appropriate.

   :mask-array (INDArray or vec of mask values),

   :mask-state (keyword), either :active or :passthrough
    - :active = apply mask to activations and errors.
    - :passthrough = feed forward the input mask (if/when necessary) but don't actually apply it.
    - Note: Masks should not be applied in all cases, depends on the network configuration

  :batch-size (int) the minibatch size to use"
  [& {:keys [layer mask-array mask-state batch-size as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :mask-array (:or (_ :guard vector?)
                            (_ :guard seq?))
           :mask-state (:or (_ :guard keyword?)
                            (_ :guard seq?))
           :batch-size (:or (_ :guard number?)
                            (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.feedForwardMaskArray ~layer (vec-or-matrix->indarray ~mask-array)
                                 (enum/value-of {:mask-state ~mask-state})
                                 (int ~batch-size)))
         :else
         (let [[l m-vec m-state b-size]
               (eval-if-code [layer seq?] [mask-array seq?]
                             [mask-state seq? keyword?]
                             [batch-size seq? number?])]
           (.feedForwardMaskArray l (vec-or-matrix->indarray m-vec)
                                  (enum/value-of {:mask-state m-state})
                                  b-size))))

(defn backprop-gradient
  "Calculate the gradient relative to the error in the next layer
   epsilon is an INDArray or vec"
  [& {:keys [layer epsilon as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :epsilon (:or (_ :guard vector?)
                         (_ :guard seq?))}]
         (obj-or-code? as-code? `(.backpropGradient ~layer (vec-or-matrix->indarray ~epsilon)))
         :else
         (let [[e l] (eval-if-code [epsilon seq?] [layer seq?])]
          (.backpropGradient l (vec-or-matrix->indarray e)))))

(defn calc-l1
  "Calculate the l1 regularization term. 0.0 if regularization is not used."
  [& {:keys [layer backprop-only-params? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :backprop-only-params? (:or (_ :guard boolean?)
                                       (_ :guard seq?))}]
         (obj-or-code? as-code? `(.calcL1 ~layer ~backprop-only-params?))
         :else
         (let [[bp-only?] (eval-if-code [backprop-only-params? seq? boolean?])]
           (.calcL1 layer bp-only?))))

(defn calc-l2
  "Calculate the l2 regularization term. 0.0 if regularization is not used."
  [& {:keys [layer backprop-only-params? as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :backprop-only-params? (:or (_ :guard boolean?)
                                       (_ :guard seq?))}]
         (obj-or-code? as-code? `(.calcL2 ~layer ~backprop-only-params?))
         :else
         (let [[bp-only?] (eval-if-code [backprop-only-params? seq? boolean?])]
           (.calcL2 layer bp-only?))))

(defn is-pretrain-layer?
  "Returns true if the layer can be trained in an unsupervised/pretrain manner (VAE, RBMs etc)"
  [layer & {:keys [as-code?]
            :or {as-code? true}}]
  (match [layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.isPretrainLayer ~layer))
         :else
         (.isPretrainLayer layer)))

(defn pre-output
  "returns the raw activations for the supplied layer

  :input (INDArray or vec), the input to the layer

  :training? (boolean), are we in training or testing mode?

  :training-mode (keyword), are we in training or testing mode?
   one of :training or :testing

  multiple layers implement this fn"
  [& {:keys [layer input training? training-mode as-code?]
      :or {as-code? true}
      :as opts}]
  (match [opts]
         [{:layer (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.preOutput ~layer (vec-or-matrix->indarray ~input) ~training?))
         [{:layer _ :input _ :training? _}]
         (let [[i t? l] (eval-if-code [input seq?] [training? seq? boolean?]
                                      [layer seq?])]
           (.preOutput l (vec-or-matrix->indarray i) t?))
         [{:layer (_ :guard seq?)
           :input (:or (_ :guard vector?)
                       (_ :guard seq?))
           :training-mode (:or (_ :guard keyword?)
                               (_ :guard seq?))}]
         (obj-or-code?
          as-code?
          `(.preOutput ~layer
                      (vec-or-matrix->indarray ~input)
                      (enum/value-of {:training-mode ~training-mode})))
         [{:layer _ :input _ :training-mode _}]
         (let [[i t-m l] (eval-if-code [input seq?]
                                       [training-mode seq? keyword?]
                                       [layer seq?])]
           (.preOutput l
                       (vec-or-matrix->indarray i)
                       (enum/value-of {:training-mode t-m})))
         [{:layer (_ :guard seq?)
           :training? (:or (_ :guard boolean?)
                           (_ :guard seq?))}]
         (obj-or-code? as-code? `(.preOutput ~layer ~training?))
         [{:layer _ :training? _}]
         (let [[t?] (eval-if-code [training? seq? boolean?])]
           (.preOutput layer t?))
         [{:layer (_ :guard seq?)
           :input (:or (_ :guard seq?)
                       (_ :guard vector?))}]
         (obj-or-code? as-code? `(.preOutput ~layer (vec-or-matrix->indarray ~input)))
         [{:layer _ :input _ }]
         (let [[i l] (eval-if-code [input seq?] [layer seq?])]
           (.preOutput l (vec-or-matrix->indarray i)))))

(defn transpose
  "Return a transposed copy of the weights/bias
  (this means reverse the number of inputs and outputs on the weights)"
  [layer & {:keys [as-code?]
            :or {as-code? true}}]
  (match [layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.transpose ~layer))
         :else
         (.transpose layer)))

(defn reset-layer-default-config
  [layer & {:keys [as-code?]
            :or {as-code? true}}]
  (match [layer]
         [(_ :guard seq?)]
         (obj-or-code? as-code? `(.resetLayerDefaultConfig ~layer))
         :else
         (.resetLayerDefaultConfig layer)))
