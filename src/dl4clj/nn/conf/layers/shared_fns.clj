(ns ^{:doc "see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/Layer.html
and https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/package-frame.html"}
    dl4clj.nn.conf.layers.shared-fns
  (:import [org.deeplearning4j.nn.conf.layers
            Layer BaseOutputLayer CenterLossOutputLayer SubsamplingLayer
            ActivationLayer AutoEncoder BasePretrainNetwork BaseRecurrentLayer
            BatchNormalization Convolution1DLayer ConvolutionLayer
            DenseLayer DropoutLayer EmbeddingLayer FeedForwardLayer
            GlobalPoolingLayer GravesBidirectionalLSTM GravesLSTM
            LocalResponseNormalization LossLayer OutputLayer RBM
            RnnOutputLayer Subsampling1DLayer ZeroPaddingLayer])
  (:require [dl4clj.nn.conf.constants :as enum]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From Layer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-l1-by-param
  "Get the L1 coefficient for the given parameter."
  [& {:keys [layer param-name]}]
  (.getL1ByParam layer param-name))

(defn get-l2-by-param
  "Get the L2 coefficient for the given parameter."
  [& {:keys [layer param-name]}]
  (.getL2ByParam layer param-name))

(defn get-learning-rate-by-param
  "Get the (initial) learning rate coefficient for the given parameter."
  [& {:keys [layer param-name]}]
  (.getLearningRateByParam layer param-name))

(defn get-output-type
  "returns the output type of the provided layer given the layer input type

  :layer-idx (int), the index of the layer within its model

  :input-type (map), the input to the cnn layer
   - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants"
  [& {:keys [layer layer-idx input-type]}]
  (.getOutputType layer layer-idx (enum/input-types input-type)))

(defn get-pre-processor-for-input-type
  "For the given type of input to this layer, what preprocessor (if any) is required?

  :input-type (map), the input to the cnn layer
   - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants"
  [& {:keys [layer input-type]}]
  (.getPreProcessorForInputType layer (enum/input-types input-type)))

(defn get-updater-by-param
  "Get the updater for the given parameter."
  [& {:keys [layer param-name]}]
  (.getUpdaterByParam layer param-name))

(defn initializer
  "returns a param initializer for this layer"
  [layer]
  (.initializer layer))

(defn instantiate
  "returns the instantiated layer

  :layer (layer), the layer set in the nn-conf

  :conf (nn-conf), a single layer neural network configuration

  :listener (coll), a collection of listeners for the layer

  :layer-idx (int), the index of the layer within the model

  :layer-param-view (INDArray or vec), the params of the layer

  :initialize-params? (boolean), do you want to initialize the params?"
  [& {:keys [layer conf listener layer-idx
             layer-param-view initialize-params?]}]
  (.instantiate layer conf listener layer-idx
                (vec-or-matrix->indarray layer-param-view)
                initialize-params?))

(defn set-n-in!
  "Set the nIn value (number of inputs, or input depth for CNNs) based on the given input type

  :input-type (map), the input to the cnn layer
   - {:convolutional {:height 1 :width 1 :depth 1}}
   - {:recurrent {:size 10}}
  - only 2 examples, see dl4clj.nn.conf.constants

  :override? (boolean), do you want to override the current n-in?

  returns the layer"
  [& {:keys [layer input-type override?]}]
  (doto layer (.setNIn (enum/input-types input-type) override?)))

(defn reset-layer-default-config
  [layer]
  (.resetLayerDefaultConfig layer))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From BaseOutputlayer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-loss-fn
  [output-layer]
  (.getLossFn output-layer))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From CenterLossOutputLayer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-alpha
  [center-loss-output-layer]
  (.getAlpha center-loss-output-layer))

(defn get-gradient-check
  [center-loss-output-layer]
  (.getGradientCheck center-loss-output-layer))

(defn get-lambda
  [center-loss-output-layer]
  (.getLambda center-loss-output-layer))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; From SubsamplingLayer
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-eps
  [subsampling-layer]
  (.getEps subsampling-layer))

(defn get-pnorm
  [subsampling-layer]
  (.getPnorm subsampling-layer))
