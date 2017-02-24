(ns dl4clj.nn.conf.builders.builders
  (:require [dl4clj.nn.conf.distribution.distribution :as distribution]
            [dl4clj.nn.conf.gradient-normalization :as g-norm]
            [dl4clj.nn.conf.learning-rate-policy :as l-rate-p]
            [dl4clj.nn.conf.activation-fns :as activation-fn]
            [dl4clj.nn.api.optimization-algorithm :as opt-algo]
            [dl4clj.nn.conf.step-fns :as step-functions]
            [dl4clj.nn.conf.updater :as updaters]
            [nd4clj.linalg.lossfunctions.loss-functions :as loss-functions]
            [dl4clj.nn.conf.updater :as updater]
            [dl4clj.nn.weights.weight-init :as w-init])
  (:import
   [org.deeplearning4j.nn.conf.layers
    Layer$Builder FeedForwardLayer$Builder ActivationLayer$Builder BaseOutputLayer$Builder
    OutputLayer$Builder RnnOutputLayer$Builder BasePretrainNetwork$Builder AutoEncoder$Builder
    RBM$Builder BaseRecurrentLayer$Builder GravesBidirectionalLSTM$Builder
    GravesLSTM$Builder BatchNormalization$Builder ConvolutionLayer$Builder
    DenseLayer$Builder EmbeddingLayer$Builder LocalResponseNormalization$Builder
    SubsamplingLayer$Builder SubsamplingLayer$PoolingType ConvolutionLayer$AlgoMode]
   [org.nd4j.linalg.activations Activation]))

(defn layer-type [opts]
  (first (keys opts)))
;; write specs to check for correct keys given a layer type
(defmulti builder
  ":layer opts:
   :activation-fn, :adam-mean-decay, :adam-var-decay, :bias-init, :bias-learning-rate,
   :dist :drop-out :epsilon, :gradient-normalization, :gradient-normalization-threshold,
   :l1 :l2 :layer-name, :learning-rate, :learning-rate-policy, :momentum, :momentum-after,
   :rho, :rms-decay, :updater, weight-init

  subclasses of :layer, FeedForwardLayer, LocalResponseNormalization, SubsamplingLayer

  FeedForwardlayer adds :n-in and :n-out
  LocalResponsenormalization adds :alpha, :beta, :k, :n
  Subsamplinglayer adds :kernel-size, :padding, :pooling-type, :stride

  subclasses of FeedFowardLayer: ActivationLayer, BaseOutputlayer, BasePretrainnetwork,
                                 BaseRecurrentlayer, BatchNormalization, Convolutionlayer,
                                 DenseLayer, EmbeddingLayer
  LocalResponseNormalization and SubSamplingLayer have no subclasses

  :activation-layer adds no extra params and has no subclasses

  :base-output-layer adds :loss-fn and has two subclasses:
  :output-layer and :rnn-output-layer, neither adds any params

  :base-pretrain-network adds :loss-fn and has two subclasses:
  :auto-encoder and :rbm
  :auto-encoder adds :corruption-level and :sparsity
  :rbm adds :hidden-unit, :vissible-unit, :k, :sparsity

  :base-recurrent-layer adds no extra params and has two subclasses:
  :graves-lstm and :graves-bidirectional-lstm which both add :forget-gate-bias-init

  :batch-normalization has no subclasses and adds:
  :beta, :decay, :eps, :gamma, :is-mini-batch, :lock-gamma-beta

  :convolutional-layer has no subclasses and adds:
  :convolution-type, :cudnn-algo-mode, :kernel-size, :padding, :stride

  :dense-layer has no subclasses and has no additional params

  :embedding-layer has no subclasses and has no additional params"

  layer-type)

(defn any-layer-builder
  "given a map of layer-type and config, generates the desired layer.  Works for
  all layer types.  Values specified in the config map will not be overwritten by nn-conf-builder.

  Params are:

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :activation-fn (keyword) one of: :cube, :elu, :hard-sigmoid, :hard-tanh, :identity,
                                   :leaky-relu :relu, :r-relu, :sigmoid, :soft-max,
                                   :soft-plus, :soft-sign, :tanh

  :adam-mean-decay (double) Mean decay rate for Adam updater

  :adam-var-decay (double) Variance decay rate for Adam updater

  :bias-init (double) Constant for bias initialization

  :bias-learning-rate (double) Bias learning rate

  :dist (map) distribution to sample initial weights from, one of:
        binomial-distribution {:binomial {:number-of-trails int :probability-of-success double}}
        normal-distribution {:normal {:mean double :std double}}
        uniform-distribution {:uniform {:lower double :upper double}}

  :drop-out (double) Dropout probability

  :epsilon (double) Epsilon value for updaters: Adagrad and Adadelta

  :gradient-normalization (keyword) gradient normalization strategy,
   These are applied on raw gradients, before the gradients are passed to the updater
   (SGD, RMSProp, Momentum, etc)
   one of: :none (default), :renormalize-l2-per-layer, :renormalize-l2-per-param-type,
           :clip-element-wise-absolute-value, :clip-l2-per-layer, :clip-l2-per-param-type
   reference: https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/GradientNormalization.html

  :gradient-normalization-threshold (double) Threshold for gradient normalization,
   only used for :clip-l2-per-layer, :clip-l2-per-param-type, :clip-element-wise-absolute-value,
   L2 threshold for first two types of clipping or absolute value threshold for the last type

  :l1 (double) L1 regularization coefficient

  :l2 (double) L2 regularization coefficient used when regularization is set to true

  :layer-name (string) Name of the layer

  :learning-rate (double) Paramter that controls the learning rate

  :learning-rate-policy (keyword) How to decay learning rate during training
   see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/LearningRatePolicy.html
   one of :none, :exponential, :inverse, :poly, :sigmoid, :step, :torch-step :schedule :score

  :learning-rate-schedule {int double} Map of the iteration to the learning rate to apply at that iteration

  :momentum (double) Momentum rate used only when the :updater is set to :nesterovs

  :momentum-after {int double} Map of the iteration to the momentum rate to apply at that iteration
   also only used when :updater is :nesterovs

  :rho (double) Ada delta coefficient

  :rms-decay (double) Decay rate for RMSProp, only applies if using :updater :RMSPROP

  :updater (keyword) Gradient updater,
   one of: :adagrad, :sgd, :adam, :adadelta, :nesterovs, :adagrad, :rmsprop, :none, :custom

  :weight-init (keyword) Weight initialization scheme
  see https://deeplearning4j.org/doc/org/deeplearning4j/nn/weights/WeightInit.html (use cases)
  one of: :distribution, :zero, :sigmoid-uniform, :uniform, :xavier, :xavier-uniform
          :xavier-fan-in, :xavier-legacy, :relu, :relu-uniform, :vi, :size, :normalized

  :loss-fn (keyword) Error measurement at output layer.  The layer types which use this
   field are: :base-output-layer, :output-layer, :rnn-output-layer
              :base-pretrain-network, :auto-encoder, :rbm
   the loss-fn opts are: :mse, :l1, :xent, :mcxent, :squared-loss,
                         :reconstruction-crossentropy, :negativeloglikelihood,
                         :cosine-proximity, :hinge, :squared-hinge, :kl-divergence,
                         :mean-absolute-error, :l2, :mean-absolute-percentage-error,
                         :mean-squared-logarithmic-error, :poisson

  :corruption-level (double) turns the autoencoder into a denoising autoencoder:
   see http://deeplearning.net/tutorial/dA.html (code examples in python) and
   http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/217

   The denoising auto-encoder is a stochastic version of the auto-encoder. Intuitively,
   a denoising auto-encoder does two things: try to encode the input (preserve the information about the input),
   and try to undo the effect of a corruption process stochastically applied to the input of the auto-encoder.
   The latter can only be done by capturing the statistical dependencies between the inputs.

  :sparsity (double), see http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity

  :hidden-unit
"
  [builder-type {:keys [n-in n-out activation-fn adam-mean-decay adam-var-decay
                        bias-init bias-learning-rate dist drop-out epsilon
                        gradient-normalization gradient-normalization-threshold
                        l1 l2 layer-name learning-rate learning-rate-policy
                        learning-rate-schedule momentum momentum-after rho
                        rms-decay updater weight-init loss-fn corruption-level
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
  (if (contains? opts :activation-fn)
    (.activation builder-type (activation-fn/value-of activation-fn)) builder-type)
  (if (contains? opts :adam-mean-decay)
    (.adamMeanDecay builder-type adam-mean-decay) builder-type)
  (if (contains? opts :adam-var-decay)
    (.adamVarDecay builder-type adam-var-decay) builder-type)
  (if (contains? opts :bias-init)
    (.biasInit builder-type bias-init) builder-type)
  (if (contains? opts :bias-learning-rate)
    (.biasLearningRate builder-type bias-learning-rate) builder-type)
  (if (contains? opts :dist)
    (.dist builder-type (if (map? dist)
               (distribution/distribution dist)
               dist)) builder-type)
  (if (contains? opts :drop-out)
    (.dropOut builder-type drop-out) builder-type)
  (if (contains? opts :epsilon)
    (.epsilon builder-type epsilon) builder-type)
  (if (contains? opts :gradient-normalization)
    (.gradientNormalization
     builder-type
     (g-norm/value-of gradient-normalization))
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
    (.learningRateDecayPolicy
     builder-type
     (l-rate-p/value-of learning-rate-policy)) builder-type)
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
    (.updater builder-type (updaters/value-of updater)) builder-type)
  (if (contains? opts :weight-init)
    (.weightInit builder-type (w-init/value-of weight-init)) builder-type)
  (if (contains? opts :loss-fn)
    (.lossFunction builder-type (loss-functions/value-of loss-fn)) builder-type)
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
  (any-layer-builder (Layer$Builder.) (:layer opts)))

(defmethod builder :feed-forward-layer [opts]
  (any-layer-builder (FeedForwardLayer$Builder.) (:feed-forward-layer opts)))

(defmethod builder :activation-layer [opts]
  (any-layer-builder (ActivationLayer$Builder.) (:activation-layer  opts)))

(defmethod builder :base-output-layer [opts]
  (any-layer-builder (BaseOutputLayer$Builder.) (:base-output-layer  opts)))

;; add :custom-output-layer-builder
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/custom/testlayers/CustomOutputLayer.Builder.html
;; add :custom-loss-function method to any-layer
;; https://deeplearning4j.org/customizelossfunction
;; jk its depreciated
(defmethod builder :output-layer [opts]
  (any-layer-builder (OutputLayer$Builder.) (:output-layer  opts)))

(defmethod builder :rnn-output-layer [opts]
  (any-layer-builder (RnnOutputLayer$Builder.) (:rnn-output-layer  opts)))

(defmethod builder :base-pretrain-network [opts]
  (any-layer-builder (BasePretrainNetwork$Builder.) (:base-pretrain-network  opts)))

(defmethod builder :auto-encoder [opts]
  (any-layer-builder (AutoEncoder$Builder.) (:auto-encoder  opts)))

(defmethod builder :rbm [opts]
  (any-layer-builder (RBM$Builder.) (:rmb  opts)))

(defmethod builder :base-recurrent-layer [opts]
  (any-layer-builder (BaseRecurrentLayer$Builder.) (:base-recurrent-layer  opts)))

(defmethod builder :graves-bidirectional-lstm [opts]
  (any-layer-builder (GravesBidirectionalLSTM$Builder.) (:graves-bidirectional-lstm  opts)))

(defmethod builder :graves-lstm [opts]
  (any-layer-builder (GravesLSTM$Builder.) (:graves-lstm  opts)))

(defmethod builder :batch-normalization [opts]
  (any-layer-builder (BatchNormalization$Builder.) (:batch-normalization  opts)))

(defmethod builder :convolutional-layer [opts]
  (any-layer-builder (ConvolutionLayer$Builder.) (:convolutional-layer  opts)))

(defmethod builder :dense-layer [opts]
  (any-layer-builder (DenseLayer$Builder.) (:dense-layer  opts)))

(defmethod builder :embedding-layer [opts]
  (any-layer-builder (EmbeddingLayer$Builder.) (:embedding-layer  opts)))

(defmethod builder :local-response-normalization [opts]
  (any-layer-builder (LocalResponseNormalization$Builder.) (:local-response-normalization  opts)))

(defmethod builder :subsampling-layer [opts]
  (any-layer-builder (SubsamplingLayer$Builder.) (:subsampling-layer  opts)))

#_(.build (builder {:graves-lstm {:l1 0.0,
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
                                  :gradient-normalization-threshold 1.0}}))
