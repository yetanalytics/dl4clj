(ns dl4clj.nn.conf.constants
  (:require [clojure.string :as s]
            [dl4clj.utils :as u])
  (:import [org.deeplearning4j.nn.conf GradientNormalization LearningRatePolicy Updater]
           [org.deeplearning4j.nn.conf.layers ConvolutionLayer$AlgoMode
            RBM$VisibleUnit RBM$HiddenUnit ConvolutionLayer$AlgoMode
            SubsamplingLayer$PoolingType]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.nd4j.linalg.activations Activation]
           [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction]
           [org.nd4j.linalg.convolution Convolution$Type]))

(defn constant-type
  [opts]
  (first (keys opts)))

(defmulti value-of constant-type)

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

(defmethod value-of :convolution-type [opts]
  (constants #(Convolution$Type/valueOf %) (:convolution-type opts)))

(defmethod value-of :cudnn-algo-mode [opts]
  (constants #(ConvolutionLayer$AlgoMode/valueOf %) (:cudnn-algo-mode opts)))

(defmethod value-of :pool-type [opts]
  (constants #(SubsamplingLayer$PoolingType/valueOf %) (:pool-type opts)))
