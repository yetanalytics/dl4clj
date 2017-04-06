(ns dl4clj.nn.conf.constants
  (:require [clojure.string :as s]
            [dl4clj.utils :as u])
  (:import [org.deeplearning4j.nn.conf GradientNormalization LearningRatePolicy
            Updater BackpropType ConvolutionMode]
           [org.deeplearning4j.nn.conf.layers ConvolutionLayer$AlgoMode
            RBM$VisibleUnit RBM$HiddenUnit
            PoolingType]
           [org.deeplearning4j.nn.conf.inputs InputType]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]
           [org.nd4j.linalg.convolution Convolution$Type]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn constant-type
  [opts]
  (first (keys opts)))

(defmulti value-of constant-type)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn heavy lifting
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn constants
  [l-type k & {:keys [activation?
                      camel?]
               :or {activation? false
                    camel? false}}]
  (let [val (name k)]
    (if camel?
      (l-type (u/camelize val true))
      (cond activation?
            (cond (s/includes? val "-")
                  (l-type (s/upper-case (s/join (s/split val #"-"))))
                  :else
                  (l-type (s/upper-case val)))
            :else
            (l-type (s/replace (s/upper-case val) "-" "_"))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn methods
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmethod value-of :activation-fn [opts]
  (constants #(Activation/valueOf %) (:activation-fn opts) :activation? true))

(defmethod value-of :gradient-normalization [opts]
  (constants #(GradientNormalization/valueOf %) (:gradient-normalization opts) :camel? true))

(defmethod value-of :learning-rate-policy [opts]
  (constants #(LearningRatePolicy/valueOf %) (:learning-rate-policy opts) :camel? true))

(defmethod value-of :updater [opts]
  (constants #(Updater/valueOf %) (:updater opts)))

(defmethod value-of :weight-init [opts]
  (constants #(WeightInit/valueOf %) (:weight-init opts)))

(defmethod value-of :loss-fn [opts]
  (constants #(LossFunctions$LossFunction/valueOf %) (:loss-fn opts)))

(defmethod value-of :hidden-unit [opts]
  (constants #(RBM$HiddenUnit/valueOf %) (:hidden-unit opts)))

(defmethod value-of :visible-unit [opts]
  (constants #(RBM$VisibleUnit/valueOf %) (:visible-unit opts)))

(defmethod value-of :convolution-mode [opts]
  (constants #(ConvolutionMode/valueOf %) (:convolution-mode opts) :camel? true))

(defmethod value-of :cudnn-algo-mode [opts]
  (constants #(ConvolutionLayer$AlgoMode/valueOf %) (:cudnn-algo-mode opts)))

(defmethod value-of :pool-type [opts]
  (constants #(PoolingType/valueOf %) (:pool-type opts)))

(defmethod value-of :backprop-type [opts]
  (if (= (:backprop-type opts) :truncated-bptt)
    (BackpropType/valueOf "TruncatedBPTT")
    (BackpropType/valueOf "Standard")))

(defmethod value-of :optimization-algorithm [opts]
  (constants #(OptimizationAlgorithm/valueOf %) (:optimization-algorithm opts)))

(defn input-types
  [opts]
  (let [{typez :type
         height :height
         width :width
         depth :depth
         size :size} (:input-type opts)]
    (cond
      (= typez :convolutional)
      (InputType/convolutional height width depth)
      (= typez :convolutional-flat)
      (InputType/convolutionalFlat height width depth)
      (= typez :feed-forward)
      (InputType/feedForward size)
      (= typez :recurrent)
      (InputType/recurrent size))))
