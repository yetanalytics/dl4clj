(ns dl4clj.nn.params.param-initializers
  (:import [org.deeplearning4j.nn.params
            BatchNormalizationParamInitializer
            CenterLossParamInitializer
            ConvolutionParamInitializer
            DefaultParamInitializer
            EmptyParamInitializer
            GravesBidirectionalLSTMParamInitializer
            GravesLSTMParamInitializer
            PretrainParamInitializer
            VariationalAutoencoderParamInitializer])
  (:require [dl4clj.utils :refer [contains-many?]]))

;; dont htink this is core. verify this

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Multimethod for calling constructors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti initializers identity)

(defmethod initializers :batch-normalization [opts]
  (BatchNormalizationParamInitializer.))

(defmethod initializers :center-loss [opts]
  (CenterLossParamInitializer.))

(defmethod initializers :convolution [opts]
  (ConvolutionParamInitializer.))

(defmethod initializers :default [opts]
  (DefaultParamInitializer.))

(defmethod initializers :empty [opts]
  (EmptyParamInitializer.))

(defmethod initializers :bidirectional-lstm [opts]
  (GravesBidirectionalLSTMParamInitializer.))

(defmethod initializers :lstm [opts]
  (GravesLSTMParamInitializer.))

(defmethod initializers :pre-train [opts]
  (PretrainParamInitializer.))

(defmethod initializers :vae [opts]
  (VariationalAutoencoderParamInitializer.))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fns using the multimethod
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-batch-norm-initializer
  []
  (initializers :batch-normalization))

(defn new-center-loss-initializer
  []
  (initializers :center-loss))

(defn new-convolution-initializer
  []
  (initializers :convolution))

(defn new-default-initializer
  []
  (initializers :default))

(defn new-empty-initializer
  []
  (initializers :empty))

(defn new-bidirectional-lstm-initializer
  []
  (initializers :bidirectional-lstm))

(defn new-lstm-initializer
  []
  (initializers :lstm))

(defn new-pre-train-initializer
  []
  (initializers :pre-train))

(defn new-vae-initializer
  []
  (initializers :vae))
