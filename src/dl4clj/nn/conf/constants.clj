(ns dl4clj.nn.conf.constants
  (:require [clojure.string :as s]
            [dl4clj.utils :refer :all])
  (:import [org.deeplearning4j.nn.conf GradientNormalization LearningRatePolicy
            Updater BackpropType ConvolutionMode]
           [org.deeplearning4j.nn.conf.layers ConvolutionLayer$AlgoMode
            RBM$VisibleUnit RBM$HiddenUnit
            PoolingType]
           [org.deeplearning4j.nn.conf.inputs InputType]
           [org.deeplearning4j.nn.api OptimizationAlgorithm MaskState
            Layer$Type Layer$TrainingMode]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]
           [org.nd4j.linalg.convolution Convolution$Type]
           [org.deeplearning4j.datasets.datavec RecordReaderMultiDataSetIterator$AlignmentMode
            SequenceRecordReaderDataSetIterator$AlignmentMode]))

;; not used anywhere yet
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/MaskState.html
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Layer.Type.html
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Layer.TrainingMode.html

;; datasets constants
;;https://deeplearning4j.org/doc/org/deeplearning4j/datasets/datavec/RecordReaderMultiDataSetIterator.AlignmentMode.html
;;https://deeplearning4j.org/doc/org/deeplearning4j/datasets/datavec/SequenceRecordReaderDataSetIterator.AlignmentMode.html

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti value-of generic-dispatching-fn)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn heavy lifting
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn constants
  "keyword to string formatting.

  dealing with camel case and what not"
  [l-type k & {:keys [activation?
                      camel?]
               :or {activation? false
                    camel? false}}]
  (let [val (name k)]
    (if camel?
      (l-type (camelize val true))
      (cond activation?
            (cond (s/includes? val "-")
                  (l-type (s/upper-case (s/join (s/split val #"-"))))
                  :else
                  (l-type (s/upper-case val)))
            :else
            (l-type (s/replace (s/upper-case val) "-" "_"))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn methods (source for all possible values in the comment)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmethod value-of :activation-fn [opts]
  (constants #(Activation/valueOf %) (:activation-fn opts) :activation? true))
;; http://nd4j.org/doc/org/nd4j/linalg/activations/Activation.html

(defmethod value-of :gradient-normalization [opts]
  (constants #(GradientNormalization/valueOf %) (:gradient-normalization opts) :camel? true))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/GradientNormalization.html

(defmethod value-of :learning-rate-policy [opts]
  (constants #(LearningRatePolicy/valueOf %) (:learning-rate-policy opts) :camel? true))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/LearningRatePolicy.html

(defmethod value-of :updater [opts]
  (constants #(Updater/valueOf %) (:updater opts)))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/Updater.html

(defmethod value-of :weight-init [opts]
  (constants #(WeightInit/valueOf %) (:weight-init opts)))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/weights/WeightInit.html

(defmethod value-of :loss-fn [opts]
  (constants #(LossFunctions$LossFunction/valueOf %) (:loss-fn opts)))
;; http://nd4j.org/doc/org/nd4j/linalg/lossfunctions/LossFunctions.LossFunction.html

(defmethod value-of :hidden-unit [opts]
  (constants #(RBM$HiddenUnit/valueOf %) (:hidden-unit opts)))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/RBM.HiddenUnit.html

(defmethod value-of :visible-unit [opts]
  (constants #(RBM$VisibleUnit/valueOf %) (:visible-unit opts)))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/RBM.VisibleUnit.html

(defmethod value-of :convolution-mode [opts]
  (constants #(ConvolutionMode/valueOf %) (:convolution-mode opts) :camel? true))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ConvolutionMode.html

(defmethod value-of :cudnn-algo-mode [opts]
  (constants #(ConvolutionLayer$AlgoMode/valueOf %) (:cudnn-algo-mode opts)))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/ConvolutionLayer.AlgoMode.html

(defmethod value-of :pool-type [opts]
  (constants #(PoolingType/valueOf %) (:pool-type opts)))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/PoolingType.html

(defmethod value-of :backprop-type [opts]
  (if (= (:backprop-type opts) :truncated-bptt)
    (BackpropType/valueOf "TruncatedBPTT")
    (BackpropType/valueOf "Standard")))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/BackpropType.html

(defmethod value-of :optimization-algorithm [opts]
  (constants #(OptimizationAlgorithm/valueOf %) (:optimization-algorithm opts)))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/OptimizationAlgorithm.html

(defmethod value-of :mask-state [opts]
  (constants #(MaskState/valueOf %) (:mask-state opts) :camel? true))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/MaskState.html

(defmethod value-of :layer-type [opts]
  (constants #(Layer$Type/valueOf %) (:layer-type opts)))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Layer.Type.html

(defmethod value-of :layer-training-mode [opts]
  (constants #(Layer$TrainingMode/valueOf %) (:layer-training-mode opts)))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/Layer.TrainingMode.html

(defn input-types
  [opts]
  (let [typez (generic-dispatching-fn opts)
        {height :height
         width :width
         depth :depth
         size :size} (typez opts)]
    (cond
      (= typez :convolutional)
      (InputType/convolutional height width depth)
      (= typez :convolutional-flat)
      (InputType/convolutionalFlat height width depth)
      (= typez :feed-forward)
      (InputType/feedForward size)
      (= typez :recurrent)
      (InputType/recurrent size))))
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/inputs/InputType.html
