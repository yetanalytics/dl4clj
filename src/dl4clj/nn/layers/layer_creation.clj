(ns dl4clj.nn.layers.layer-creation
  (:import [org.deeplearning4j.nn.layers ActivationLayer DropoutLayer
            FrozenLayer LossLayer OutputLayer]
           [org.deeplearning4j.nn.layers.training CenterLossOutputLayer]
           [org.deeplearning4j.nn.layers.convolution.subsampling Subsampling1DLayer
            SubsamplingLayer]
           [org.deeplearning4j.nn.layers.convolution Convolution1DLayer
            ConvolutionLayer ZeroPaddingLayer]
           [org.deeplearning4j.nn.layers.feedforward.autoencoder AutoEncoder]
           [org.deeplearning4j.nn.layers.feedforward.dense DenseLayer]
           [org.deeplearning4j.nn.layers.feedforward.embedding EmbeddingLayer]
           [org.deeplearning4j.nn.layers.feedforward.rbm RBM]
           [org.deeplearning4j.nn.layers.pooling GlobalPoolingLayer]
           [org.deeplearning4j.nn.layers.normalization BatchNormalization
            LocalResponseNormalization]
           [org.deeplearning4j.nn.layers.recurrent
            GravesBidirectionalLSTM GravesLSTM RnnOutputLayer FwdPassReturn]
           [org.deeplearning4j.nn.layers.variational VariationalAutoencoder])
  (:require [dl4clj.nn.api.multi-layer-conf :refer [to-json]]
            [cheshire.core :refer [decode-strict]]
            [dl4clj.utils :refer [generic-dispatching-fn camel-to-dashed]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn layer-type
  [opts]
  (let [single-layer-nn-conf (:nn-conf opts)]
    (-> (to-json single-layer-nn-conf)
        (decode-strict true)
        :layer
        generic-dispatching-fn
        camel-to-dashed)))

(defmulti create-layer layer-type)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn methods
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmethod create-layer :activation [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (ActivationLayer. nn-conf input)
      (ActivationLayer. nn-conf))))

(defmethod create-layer :dropout [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (DropoutLayer. nn-conf input)
      (DropoutLayer. nn-conf))))

(defmethod create-layer :loss [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
      (if (contains? opts :input)
        (LossLayer. nn-conf input)
        (LossLayer. nn-conf))))

(defmethod create-layer :output [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (OutputLayer. nn-conf input)
      (OutputLayer. nn-conf))))

(defmethod create-layer :center-loss-output-layer [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (CenterLossOutputLayer. nn-conf input)
      (CenterLossOutputLayer. nn-conf))))

(defmethod create-layer :subsampling1d [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (Subsampling1DLayer. nn-conf input)
      (Subsampling1DLayer. nn-conf))))

(defmethod create-layer :subsampling [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (SubsamplingLayer. nn-conf input)
      (SubsamplingLayer. nn-conf))))

(defmethod create-layer :convolution1d [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (Convolution1DLayer. nn-conf input)
      (Convolution1DLayer. nn-conf))))

(defmethod create-layer :convolution [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (ConvolutionLayer. nn-conf input)
      (ConvolutionLayer. nn-conf))))

(defmethod create-layer :zero-padding [opts]
  (let [{nn-conf :nn-conf} opts]
    (ZeroPaddingLayer. nn-conf)))

(defmethod create-layer :auto-encoder [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (AutoEncoder. nn-conf input)
      (AutoEncoder. nn-conf))))

(defmethod create-layer :dense [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (DenseLayer. nn-conf input)
      (DenseLayer. nn-conf))))

(defmethod create-layer :embedding [opts]
  (let [{nn-conf :nn-conf} opts]
    (EmbeddingLayer. nn-conf)))

(defmethod create-layer :rbm [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (RBM. nn-conf input)
      (RBM. nn-conf))))

(defmethod create-layer :global-pooling [opts]
  (let [{nn-conf :nn-conf} opts]
    (GlobalPoolingLayer. nn-conf)))

(defmethod create-layer :batch-normalization [opts]
  (let [{nn-conf :nn-conf} opts]
    (BatchNormalization. nn-conf)))

(defmethod create-layer :local-response-normalization [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (LocalResponseNormalization. nn-conf input)
      (LocalResponseNormalization. nn-conf))))

(defmethod create-layer :graves-bidirectional-lstm [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (GravesBidirectionalLSTM. nn-conf input)
      (GravesBidirectionalLSTM. nn-conf))))

(defmethod create-layer :graves-lstm [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (GravesLSTM. nn-conf input)
      (GravesLSTM. nn-conf))))

(defmethod create-layer :rnnoutput [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (if (contains? opts :input)
      (RnnOutputLayer. nn-conf input)
      (RnnOutputLayer. nn-conf))))

(defmethod create-layer :variational-autoencoder [opts]
  (let [{nn-conf :nn-conf} opts]
    (VariationalAutoencoder. nn-conf)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-layer
  "creates a new layer from 2 args
   1) a nn configuration containing a single layer and optionaly some input
    - type of layer created is based upon the layer within the nn-conf

   2) some input values (optional)

  layers created by this fn are for testing what layers best fit your data
   - they are ment for experimentation not implementation
   - their implementation happens automatically when you follow the standard
     model building process (when you init the multi-layer-network)
     - ie. nn-conf -> multi-layer-conf -> multi-layer-network -> init network

  :nn-conf (nn-conf), a nn-conf with a single activation layer
   - ie. (nn-conf-builder ... :layer (activation-layer-builder ...))

  :input (INDArray) input values

  for information about a layer,
  see:https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Model.html
   - all known implementing classes has links to the layer classes implemented here
     - except for most base level layers ie. BaseOutputLayer"
  [& {:keys [nn-conf input]
      :as opts}]
  (assert (contains? opts :nn-conf) "you must supply a neural network configuration")
  (create-layer opts))

(defn new-frozen-layer
  "creates a new frozen layer from an existing layer"
  [layer]
  (FrozenLayer. layer))

(defn new-foward-pass-return
  "a helper object that does collects the results of one forward pass through
   a recurrent network
    - not sure how to properly use this, it was added by the community

  see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/recurrent/FwdPassReturn.html"
  []
  (FwdPassReturn.))
