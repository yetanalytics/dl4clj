(ns dl4clj.nn.layer-creation
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
           [org.deeplearning4j.nn.layers.variational VariationalAutoencoder]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder])
  (:require [dl4clj.nn.api.multi-layer-conf :refer [to-json]]
            [cheshire.core :refer [decode-strict]]
            [clojure.core.match :refer [match]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]
            [dl4clj.utils :refer [generic-dispatching-fn camel-to-dashed obj-or-code?]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn layer-type
  [opts]
  (let [single-layer-nn-conf (:nn-conf opts)]
    (if (seq? single-layer-nn-conf)
      (-> (to-json :multi-layer-conf (eval single-layer-nn-conf))
          (decode-strict true)
          :layer
          generic-dispatching-fn
          camel-to-dashed)
      (-> (to-json :multi-layer-conf single-layer-nn-conf)
          (decode-strict true)
          :layer
          generic-dispatching-fn
          camel-to-dashed))))

(defmulti create-layer layer-type)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn methods
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmethod create-layer :activation [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(ActivationLayer. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (ActivationLayer. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(ActivationLayer. ~nn-conf)
           :else
           (ActivationLayer. nn-conf))))

(defmethod create-layer :dropout [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(DropoutLayer. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (DropoutLayer. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(DropoutLayer. ~nn-conf)
           :else
           (DropoutLayer. nn-conf))))

(defmethod create-layer :loss [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(LossLayer. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (LossLayer. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(LossLayer. ~nn-conf)
           :else
           (LossLayer. nn-conf))))

(defmethod create-layer :output [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(OutputLayer. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (OutputLayer. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(OutputLayer. ~nn-conf)
           :else
           (OutputLayer. nn-conf))))

(defmethod create-layer :center-loss-output-layer [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(CenterLossOutputLayer. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (CenterLossOutputLayer. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(CenterLossOutputLayer. ~nn-conf)
           :else
           (CenterLossOutputLayer. nn-conf))))

(defmethod create-layer :subsampling1d [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(Subsampling1DLayer. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (Subsampling1DLayer. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(Subsampling1DLayer. ~nn-conf)
           :else
           (Subsampling1DLayer. nn-conf))))

(defmethod create-layer :subsampling [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(SubsamplingLayer. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (SubsamplingLayer. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(SubsamplingLayer. ~nn-conf)
           :else
           (SubsamplingLayer. nn-conf))))

(defmethod create-layer :convolution1d [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(Convolution1DLayer. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (Convolution1DLayer. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(Convolution1DLayer. ~nn-conf)
           :else
           (Convolution1DLayer. nn-conf))))

(defmethod create-layer :convolution [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(ConvolutionLayer. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (ConvolutionLayer. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(ConvolutionLayer. ~nn-conf)
           :else
           (ConvolutionLayer. nn-conf))))

(defmethod create-layer :zero-padding [opts]
  (let [{nn-conf :nn-conf} opts]
    (if (seq? nn-conf)
      `(ZeroPaddingLayer. ~nn-conf)
      (ZeroPaddingLayer. nn-conf))))

(defmethod create-layer :auto-encoder [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(AutoEncoder. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (AutoEncoder. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(AutoEncoder. ~nn-conf)
           :else
           (AutoEncoder. nn-conf))))

(defmethod create-layer :dense [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(DenseLayer. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (DenseLayer. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(DenseLayer. ~nn-conf)
           :else
           (DenseLayer. nn-conf))))

(defmethod create-layer :embedding [opts]
  (let [{nn-conf :nn-conf} opts]
    (if (seq? nn-conf)
      `(EmbeddingLayer. ~nn-conf)
      (EmbeddingLayer. nn-conf))))

(defmethod create-layer :rbm [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(RBM. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (RBM. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(RBM. ~nn-conf)
           :else
           (RBM. nn-conf))))

(defmethod create-layer :global-pooling [opts]
  (let [{nn-conf :nn-conf} opts]
    (if (seq? nn-conf)
      `(GlobalPoolingLayer. ~nn-conf)
      (GlobalPoolingLayer. nn-conf))))

(defmethod create-layer :batch-normalization [opts]
  (let [{nn-conf :nn-conf} opts]
    (if (seq? nn-conf)
      `(BatchNormalization. ~nn-conf)
      (BatchNormalization. nn-conf))))

(defmethod create-layer :local-response-normalization [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(LocalResponseNormalization. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (LocalResponseNormalization. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(LocalResponseNormalization. ~nn-conf)
           :else
           (LocalResponseNormalization. nn-conf))))

(defmethod create-layer :graves-bidirectional-lstm [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(GravesBidirectionalLSTM. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (GravesBidirectionalLSTM. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(GravesBidirectionalLSTM. ~nn-conf)
           :else
           (GravesBidirectionalLSTM. nn-conf))))

(defmethod create-layer :graves-lstm [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(GravesLSTM. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (GravesLSTM. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(GravesLSTM. ~nn-conf)
           :else
           (GravesLSTM. nn-conf))))

(defmethod create-layer :rnnoutput [opts]
  (let [{nn-conf :nn-conf
         input :input} opts]
    (match [opts]
           [{:nn-conf (_ :guard seq?)
             :input (:or (_ :guard vector?)
                         (_ :guard seq?))}]
           `(RnnOutputLayer. ~nn-conf (vec-or-matrix->indarray ~input))
           [{:nn-conf _
             :input _}]
           (RnnOutputLayer. nn-conf (vec-or-matrix->indarray input))
           [{:nn-conf (_ :guard seq?)}]
           `(RnnOutputLayer. ~nn-conf)
           :else
           (RnnOutputLayer. nn-conf))))

(defmethod create-layer :variational-autoencoder [opts]
  (let [{nn-conf :nn-conf} opts]
    (if (seq? nn-conf)
      `(VariationalAutoencoder. ~nn-conf)
      (VariationalAutoencoder. nn-conf))))

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
  [& {:keys [nn-conf input as-code?]
      :or {as-code? true}
      :as opts}]
  (assert (contains? opts :nn-conf) "you must supply a neural network configuration")
  (if (not (seq? nn-conf))
    (create-layer opts)
    (obj-or-code? as-code? (create-layer opts))))

(defn new-frozen-layer
  "creates a new frozen layer from an existing layer"
  [& {:keys [layer as-code?]
      :or {as-code? true}}]
  (if (seq? layer)
    (obj-or-code? as-code? `(FrozenLayer. ~layer))
    (FrozenLayer. layer)))

(defn new-foward-pass-return
  "a helper object that does collects the results of one forward pass through
   a recurrent network
    - not sure how to properly use this, it was added by the community
  see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/recurrent/FwdPassReturn.html"
  []
  (FwdPassReturn.))
