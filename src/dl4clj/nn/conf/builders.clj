(ns dl4clj.nn.conf.builders
  (:require [dl4clj.nn.conf.distribution.distribution :refer (distribution)]
            [dl4clj.nn.conf.gradient-normalization :as gradient-normalization]
            [nd4clj.linalg.lossfunctions.loss-functions :as loss-functions]
            [dl4clj.nn.conf.updater :as updater]
            [dl4clj.nn.weights.weight-init :as weight-init])
  (:import
   [org.deeplearning4j.nn.conf MultiLayerConfiguration MultiLayerConfiguration$Builder]
   [org.deeplearning4j.nn.conf NeuralNetConfiguration$ListBuilder NeuralNetConfiguration$Builder]
   [org.deeplearning4j.nn.conf.layers
    Layer$Builder FeedForwardLayer$Builder ActivationLayer$Builder BaseOutputLayer$Builder
    OutputLayer$Builder RnnOutputLayer$Builder BasePretrainNetwork$Builder AutoEncoder$Builder
    RBM$Builder BaseRecurrentLayer$Builder GravesBidirectionalLSTM$Builder
    GravesLSTM$Builder BatchNormalization$Builder ConvolutionLayer$Builder
    DenseLayer$Builder EmbeddingLayer$Builder LocalResponseNormalization$Builder
    SubsamplingLayer$Builder SubsamplingLayer$PoolingType ConvolutionLayer$AlgoMode]))

(defn layer-type [opts]
  (first (keys (:layers opts))))

(defmulti builder layer-type)

#_(builder {:layers {:graves-lstm {:l1 0.0,
                                   :drop-out 0.0,
                                   :dist {:uniform {:lower -0.08, :upper 0.08}},
                                   :rho 0.0,
                                   :forget-gate-bias-init 1.0,
                                   :activation :tanh,
                                   ;;   :learning-rate-after {},
                                   :gradient-normalization "None",
                                   :weight-init "DISTRIBUTION",
                                   :n-out 100,
                                   :adam-var-decay 0.999,
                                   :bias-init 0.0,
                                   :lr-score-based-decay 0.0,
                                   :momentum-after {},
                                   :l2 0.001,
                                   :updater "RMSPROP",
                                   :momentum 0.5,
                                   :layer-name "genisys",
                                   :n-in 50,
                                   :learning-rate 0.1,
                                   :adam-mean-decay 0.9,
                                   :rms-decay 0.95,
                                   :gradient-normalization-threshold 1.0}}})
(defn any-layer-builder
  [builder-type {:keys [n-in
                        n-out
                        activation-function ;; Layer activation function (String)
                        adam-mean-decay ;; Mean decay rate for Adam updater (double)
                        adam-var-decay ;; Variance decay rate for Adam updater (double)
                        bias-init ;; (double)
                        bias-learning-rate
                        dist ;; Distribution to sample initial weights from (Distribution or map)
                        drop-out ;; (double)
                        epsilon
                        gradient-normalization ;; Gradient normalization strategy
                        ;; (one of (dl4clj.nn.conf.gradient-normalization/values))
                        gradient-normalization-threshold ;; Threshold for gradient normalization
                        ;; only used for :clip-l2-per-layer, :clip-l2-per-param-type
                        ;; and clip-element-wise-absolute-value: L2 threshold for first two types of clipping, or absolute
                        ;; value threshold for last type of clipping
                        l1 ;; L1 regularization coefficient (double)
                        l2 ;; L2 regularization coefficient (double)
                        layer-name ;; name of the layer (string)
                        learning-rate ;; (double)
                        learning-rate-policy
                        learning-rate-schedule ;; Learning rate schedule (java.util.Map<java.lang.Integer,java.lang.Double>)
                        momentum ;; Momentum rate (double)
                        momentum-after ;; Momentum schedule. (java.util.Map<java.lang.Integer,java.lang.Double>)
                        rho ;; Ada delta coefficient (double)
                        rms-decay ;; Decay rate for RMSProp (double)
                        updater ;; Gradient updater  (one of (dl4clj.nn.conf.updater/values))
                        weight-init ;; Weight initialization scheme
                        ;; (one of (dl4clj.nn.weights.weight-init/values))
                        loss-fn
                        corruption-level
                        sparsity
                        hidden-unit
                        vissible-unit
                        k
                        forget-gate-bias-init
                        beta
                        decay
                        eps
                        gamma
                        is-mini-batch
                        lock-gamma-beta
                        kernel-size
                        stride
                        padding
                        convolution-type
                        cudnn-algo-mode
                        alpha
                        n
                        pooling-type]
                 :or {}
                 :as opts}]
  (if (contains? opts :n-in)
    (.nIn builder-type n-in) builder-type)
  (if (contains? opts :n-out)
    (.nOut builder-type n-out) builder-type)
  (if (contains? opts :activation-function)
    (.activation builder-type activation-function) builder-type)
  (if (contains? opts :adam-mean-decay)
    (.adamMeanDecay builder-type adam-mean-decay) builder-type)
  (if (contains? opts :adam-var-decay)
    (.adamVarDecay builder-type adam-var-decay) builder-type)
  (if (contains? opts :bias-init)
    (.biasInit builder-type bias-init) builder-type)
  (if (contains? opts :bias-learning-rate)
    (.biasLearningRate builder-type bias-learning-rate) builder-type)
  (if (contains? opts :dist) ;; could be a map
    (if (map? dist)
      (.dist builder-type (distribution dist))
      (.dist builder-type dist))
    builder-type)
  (if (contains? opts :drop-out)
    (.dropOut builder-type drop-out) builder-type)
  (if (contains? opts :epsilon)
    (.epsilon builder-type epsilon) builder-type)
  (if (contains? opts :gradient-normalization)
    (.gradientNormalization
     builder-type
     (gradient-normalization/value-of gradient-normalization))
    builder-type)
  (if (contains? opts :gradient-normalization-threshold)
    (.gradientNormalizationThreshold builder-type gradient-normalization-threshold) builder-type)
  (if (contains? opts :l1)
    (.l1 builder-type l1) builder-type)
  (if (contains? opts :l2)
    (.l2 builder-type l2) builder-type)
  (if (contains? opts :layer-name)
    (.name builder-type layer-name) builder-type)
  (if (contains? opts :learning-rate)
    (.learningRate builder-type learning-rate) builder-type)
  (if (contains? opts :learning-rate-policy)
    (.learningRateDecayPolicy builder-type learning-rate-policy) builder-type)
  (if (contains? opts :learning-rate-schedule)
    (.learningRateSchedule builder-type learning-rate-schedule) builder-type)
  (if (contains? opts :momentum)
    (.momentum builder-type momentum) builder-type)
  (if (contains? opts :momentum-after)
    (.momentumAfter builder-type momentum-after) builder-type)
  (if (contains? opts :rho)
    (.rho builder-type rho) builder-type)
  (if (contains? opts :rms-decay)
    (.rmsDecay builder-type rms-decay) builder-type)
  (if (contains? opts :updater)
    (.updater builder-type (updater/value-of updater)) builder-type)
  (if (contains? opts :weight-init)
    (.weightInit builder-type (weight-init/value-of weight-init)) builder-type)
  (if (contains? opts :loss-fn)
    (.lossFunction builder-type (loss-functions/value-of loss-function)) builder-type)
  (if (contains? opts :corruption-level)
    (.corruptionLevel builder-type corruption-level) builder-type)
  (if (contains? opts :sparsity)
    (.sparsity builder-type sparsity) builder-type)
  (if (contains? opts :hidden-unit)
    (.hiddenUnit builder-type hidden-unit) builder-type)
  (if (contains? opts :vissible-unit)
    (.visibleUnit builder-type vissible-unit) builder-type)
  (if (contains? opts :k)
    (.k builder-type k) builder-type)
  (if (contains? opts :forget-gate-bias-init)
    (.forgetGateBiasInit builder-type forget-gate-bias-init) builder-type)
  (if (contains? opts :beta)
    (.beta builder-type beta) builder-type)
  (if (contains? opts :decay)
    (.decay builder-type decay) builder-type)
  (if (contains? opts :eps)
    (.eps builder-type eps) builder-type)
  (if (contains? opts :gamma)
    (.gamma builder-type gamma) builder-type)
  (if (and (contains? opts :is-mini-batch) (true? is-mini-batch))
    (.isMiniBatch builder-type) builder-type)
  (if (contains? opts :lock-gamma-beta)
    (.lockGammaBeta builder-type lock-gamma-beta) builder-type)
  (if (contains? opts :kernel-size)
    (.kernelSize builder-type kernel-size) builder-type)
  (if (contains? opts :stride)
    (.stride builder-type stride) builder-type)
  (if (contains? opts :padding)
    (.padding builder-type padding) builder-type)
  (if (contains? opts :convolution-type)
    (.convolutionType builder-type convolution-type) builder-type)
  (if (contains? opts :cudnn-algo-mode)
    (.cudnnAlgoMode builder-type cudnn-algo-mode) builder-type)
  (if (contains? opts :alpha)
    (.alpha builder-type alpha) builder-type)
  (if (contains? opts :n)
    (.n builder-type n) builder-type)
  (if (contains? opts :pooling-type)
    (.poolingType builder-type pooling-type) builder-type)
  (.build builder-type))

(defmethod builder :layer [opts]
  (any-layer-builder (Layer$Builder.) (:layer (:layers opts))))

(defmethod builder :feed-forward-layer [opts]
  (any-layer-builder (FeedForwardLayer$Builder.) (:feed-forward-layer (:layers opts))))

(defmethod builder :activation-layer [opts]
  (any-layer-builder (ActivationLayer$Builder.) (:activation-layer (:layers opts))))

(defmethod builder :base-output-layer [opts]
  (any-layer-builder (BaseOutputLayer$Builder.) (:base-output-layer (:layers opts))))

(defmethod builder :output-layer [opts]
  (any-layer-builder (OutputLayer$Builder.) (:output-layer (:layers opts))))

(defmethod builder :rnn-output-layer [opts]
  (any-layer-builder (RnnOutputLayer$Builder.) (:rnn-output-layer (:layers opts))))

(defmethod builder :base-pretrain-network [opts]
  (any-layer-builder (BasePretrainNetwork$Builder.) (:base-pretrain-network (:layers opts))))

(defmethod builder :auto-encoder [opts]
  (any-layer-builder (AutoEncoder$Builder.) (:auto-encoder (:layers opts))))

(defmethod builder :rbm [opts]
  (any-layer-builder (RBM$Builder.) (:rmb (:layers opts))))

(defmethod builder :base-recurrent-layer [opts]
  (any-layer-builder (BaseRecurrentLayer$Builder.) (:base-recurrent-layer (:layers opts))))

(defmethod builder :graves-bidirectional-lstm [opts]
  (any-layer-builder (GravesBidirectionalLSTM$Builder.) (:graves-bidirectional-lstm (:layers opts))))

(defmethod builder :graves-lstm [opts]
  (any-layer-builder (GravesLSTM$Builder.) (:graves-lstm (:layers opts))))

(defmethod builder :batch-normalization [opts]
  (any-layer-builder (BatchNormalization$Builder.) (:batch-normalization (:layers opts))))

(defmethod builder :convolutional-layer [opts]
  (any-layer-builder (ConvolutionLayer$Builder.) (:convolutional-layer (:layers opts))))

(defmethod builder :dense-layer [opts]
  (any-layer-builder (DenseLayer$Builder.) (:dense-layer (:layers opts))))

(defmethod builder :embedding-layer [opts]
  (any-layer-builder (EmbeddingLayer$Builder.) (:embedding-layer (:layers opts))))

(defmethod builder :local-response-normalization [opts]
  (any-layer-builder (LocalResponseNormalization$Builder.) (:local-response-normalization (:layers opts))))

(defmethod builder :subsampling-layer [opts]
  (any-layer-builder (SubsamplingLayer$Builder.) (:subsampling-layer (:layers opts))))
