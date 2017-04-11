(ns ^{:doc "see https://deeplearning4j.org/glossary for param descriptions"}
    dl4clj.nn.conf.builders.builders
  (:require [dl4clj.nn.conf.distribution.distribution :as distribution]
            [dl4clj.nn.conf.constants :as constants]
            [dl4clj.nn.conf.utils :as u])
  (:import
   [org.deeplearning4j.nn.conf.layers ActivationLayer$Builder
    OutputLayer$Builder RnnOutputLayer$Builder AutoEncoder$Builder
    RBM$Builder GravesBidirectionalLSTM$Builder GravesLSTM$Builder
    BatchNormalization$Builder ConvolutionLayer$Builder DenseLayer$Builder
    EmbeddingLayer$Builder LocalResponseNormalization$Builder
    SubsamplingLayer$Builder LossLayer$Builder CenterLossOutputLayer$Builder
    Convolution1DLayer$Builder DropoutLayer$Builder GlobalPoolingLayer$Builder
    Layer$Builder Subsampling1DLayer$Builder ZeroPaddingLayer$Builder]
   [org.deeplearning4j.nn.conf.layers.variational VariationalAutoencoder$Builder]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn layer-type
  "dispatch fn for builder"
  [opts]
  (first (keys opts)))

(defmulti builder
  "multimethod that builds a layer based on the supplied type and opts"
  layer-type)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn heavy lifting
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; implement
;; https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/BasePretrainNetwork.Builder.html
(defn any-layer-builder
  "creates any type of layer given a builder and param map

  params shared between all layers:

  :activation-fn (keyword) one of: :cube, :elu, :hard-sigmoid, :hard-tanh, :identity,
                                   :leaky-relu :relu, :r-relu, :sigmoid, :soft-max,
                                   :soft-plus, :soft-sign, :tanh, :rational-tanh

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

  :gradient-normalization-threshold (double) Threshold for gradient normalization,
   only used for :clip-l2-per-layer, :clip-l2-per-param-type, :clip-element-wise-absolute-value,
   L2 threshold for first two types of clipping or absolute value threshold for the last type

  :l1 (double) L1 regularization coefficient

  :l2 (double) L2 regularization coefficient used when regularization is set to true

  :l1-bias (double) L1 regularization coefficient for the bias. Default: 0.

  :l2-bias (double) L2 regularization coefficient for the bias. Default: 0.

  :layer-name (string) Name of the layer

  :learning-rate (double) Paramter that controls the learning rate

  :learning-rate-policy (keyword) How to decay learning rate during training
   one of :none, :exponential, :inverse, :poly, :sigmoid, :step, :torch-step :schedule :score

  :learning-rate-schedule {int double} map of iteration to the learning rate

  :momentum (double) Momentum rate used only when the :updater is set to :nesterovs

  :momentum-after {int double} Map of the iteration to the momentum rate to apply at that iteration
   also only used when :updater is :nesterovs

  :rho (double) Ada delta coefficient

  :rms-decay (double) Decay rate for RMSProp, only applies if using :updater :RMSPROP

  :updater (keyword) Gradient updater,
   one of: :adagrad, :sgd, :adam, :adadelta, :nesterovs, :rmsprop, :none, :custom

  :weight-init (keyword) Weight initialization scheme
  one of: :distribution, :zero, :sigmoid-uniform, :uniform, :xavier, :xavier-uniform
          :xavier-fan-in, :xavier-legacy, :relu, :relu-uniform, :vi, :size, :normalized"

  [builder-type {:keys [activation-fn adam-mean-decay adam-var-decay
                        bias-init bias-learning-rate dist drop-out epsilon
                        gradient-normalization gradient-normalization-threshold
                        l1 l2 layer-name learning-rate learning-rate-policy
                        learning-rate-schedule momentum momentum-after rho
                        rms-decay updater weight-init n-in n-out loss-fn corruption-level
                        sparsity hidden-unit visible-unit k forget-gate-bias-init
                        beta decay eps gamma is-mini-batch lock-gamma-beta
                        kernel-size stride padding cudnn-algo-mode
                        alpha n pooling-type decoder-layer-sizes
                        encoder-layer-sizes num-samples pzx-activation-function
                        gradient-check lambda collapse-dimensions pnorm
                        pooling-dimensions eps convolution-mode l1-bias l2-bias
                        pre-train-iterations gate-activation-fn visible-bias-init
                        #_pzx-activation-fn
                        ;; needs to be implemented
                        reconstruction-distribution
                        ]
                 :or {}
                 :as opts}]
  (.build
   (cond-> builder-type
     (contains? opts :activation-fn) (.activation (constants/value-of {:activation-fn activation-fn}))
     (contains? opts :adam-mean-decay) (.adamMeanDecay adam-mean-decay)
     (contains? opts :adam-var-decay) (.adamVarDecay adam-var-decay)
     (contains? opts :pre-train-iterations) (.preTrainIterations pre-train-iterations)
     (contains? opts :gate-activation-fn) (.gateActivationFunction (constants/value-of
                                                                    {:activation-fn gate-activation-fn}))
     (contains? opts :bias-init) (.biasInit bias-init)
     (contains? opts :bias-learning-rate) (.biasLearningRate bias-learning-rate)
     (contains? opts :dist) (.dist (if (map? dist) (distribution/distribution dist)
                                       dist))
     (contains? opts :l1-bias) (.l1Bias l1-bias)
     (contains? opts :visible-bias-init) (.visibleBiasInit visible-bias-init)
     (contains? opts :l2-bias) (.l2Bias l2-bias)
     (contains? opts :drop-out) (.dropOut drop-out)
     (contains? opts :epsilon) (.epsilon epsilon)
     (contains? opts :gradient-normalization) (.gradientNormalization
                                               (constants/value-of
                                                {:gradient-normalization gradient-normalization}))
     (contains? opts :gradient-normalization-threshold) (.gradientNormalizationThreshold
                                                         gradient-normalization-threshold)
     (contains? opts :eps) (.eps eps)
     (contains? opts :l1) (.l1 l1)
     (contains? opts :l2) (.l2 l2)
     (contains? opts :layer-name) (.name layer-name)
     (contains? opts :learning-rate) (.learningRate learning-rate)
     (contains? opts :learning-rate-policy) (.learningRateDecayPolicy
                                             (constants/value-of
                                              {:learning-rate-policy
                                               learning-rate-policy}))
     (contains? opts :pooling-dimensions) (.poolingDimensions pooling-dimensions)
     (contains? opts :learning-rate-schedule) (.learningRateSchedule learning-rate-schedule)
     (contains? opts :momentum) (.momentum momentum)
     (contains? opts :momentum-after) (.momentumAfter momentum-after)
     (contains? opts :rho) (.rho rho)
     (contains? opts :rms-decay) (.rmsDecay rms-decay)
     (contains? opts :updater) (.updater (constants/value-of {:updater updater}))
     (contains? opts :weight-init) (.weightInit (constants/value-of
                                                 {:weight-init weight-init}))
     (contains? opts :n-in) (.nIn n-in)
     (contains? opts :n-out) (.nOut n-out)
     (contains? opts :loss-fn) (.lossFunction (constants/value-of {:loss-fn loss-fn}))
     (contains? opts :corruption-level) (.corruptionLevel corruption-level)
     (contains? opts :sparsity) (.sparsity sparsity)
     (contains? opts :visible-unit) (.visibleUnit (constants/value-of
                                                   {:visible-unit visible-unit}))
     (contains? opts :hidden-unit) (.hiddenUnit (constants/value-of
                                                 {:hidden-unit hidden-unit}))
     (contains? opts :k) (.k k)
     (contains? opts :forget-gate-bias-init) (.forgetGateBiasInit forget-gate-bias-init)
     (contains? opts :beta) (.beta beta)
     (contains? opts :decay) (.decay decay)
     (contains? opts :gamma) (.gamma gamma)
     (contains? opts :is-mini-batch) (.isMiniBatch is-mini-batch)
     (contains? opts :lock-gamma-beta) (.lockGammaBeta lock-gamma-beta)
     (contains? opts :kernel-size) (.kernelSize kernel-size)
     (contains? opts :stride) (.stride stride)
     (contains? opts :padding) (.padding padding)
     (contains? opts :convolution-mode) (.convolutionMode (constants/value-of
                                                           {:convolution-mode
                                                            convolution-mode}))
     (contains? opts :cudnn-algo-mode) (.cudnnAlgoMode (constants/value-of
                                                        {:cudnn-algo-mode
                                                         cudnn-algo-mode}))
     (contains? opts :pnorm) (.pnorm pnorm)
     (contains? opts :alpha) (.alpha alpha)
     (contains? opts :n) (.n n)
     (contains? opts :pooling-type) (.poolingType (constants/value-of
                                                   {:pool-type pooling-type}))
     (contains? opts :decoder-layer-sizes) (.decoderLayerSizes decoder-layer-sizes)
     (contains? opts :encoder-layer-sizes) (.encoderLayerSizes encoder-layer-sizes)
     (contains? opts :num-samples) (.numSamples num-samples)
     (contains? opts :gradient-check) (.gradientCheck gradient-check)
     (contains? opts :lambda) (.lambda lambda)
     (contains? opts :collapse-dimensions) (.collapseDimensions collapse-dimensions)
     (contains? opts :pzx-activation-function) (.pzxActivationFunction
                                                (constants/value-of
                                                 {:activation-fn pzx-activation-function}))
     )))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi fn methods
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmethod builder :activation-layer [opts]
  (any-layer-builder (ActivationLayer$Builder.) (:activation-layer opts)))

(defmethod builder :output-layer [opts]
  (any-layer-builder (OutputLayer$Builder.) (:output-layer opts)))

(defmethod builder :rnn-output-layer [opts]
  (any-layer-builder (RnnOutputLayer$Builder.) (:rnn-output-layer opts)))

(defmethod builder :auto-encoder [opts]
  (any-layer-builder (AutoEncoder$Builder.) (:auto-encoder opts)))

(defmethod builder :rbm [opts]
  (any-layer-builder (RBM$Builder.) (:rbm opts)))

(defmethod builder :graves-bidirectional-lstm [opts]
  (any-layer-builder (GravesBidirectionalLSTM$Builder.) (:graves-bidirectional-lstm opts)))

(defmethod builder :graves-lstm [opts]
  (any-layer-builder (GravesLSTM$Builder.) (:graves-lstm opts)))

(defmethod builder :batch-normalization [opts]
  (any-layer-builder (BatchNormalization$Builder.) (:batch-normalization opts)))

(defmethod builder :convolutional-layer [opts]
  (any-layer-builder (ConvolutionLayer$Builder.) (:convolutional-layer opts)))

(defmethod builder :dense-layer [opts]
  (any-layer-builder (DenseLayer$Builder.) (:dense-layer opts)))

(defmethod builder :embedding-layer [opts]
  (any-layer-builder (EmbeddingLayer$Builder.) (:embedding-layer opts)))

(defmethod builder :local-response-normalization [opts]
  (any-layer-builder (LocalResponseNormalization$Builder.) (:local-response-normalization opts)))

(defmethod builder :subsampling-layer [opts]
  (any-layer-builder (SubsamplingLayer$Builder.) (:subsampling-layer opts)))

(defmethod builder :variational-auto-encoder [opts]
  (any-layer-builder (VariationalAutoencoder$Builder.) (:variational-auto-encoder opts)))

(defmethod builder :loss-layer [opts]
  (any-layer-builder (LossLayer$Builder.) (:loss-layer opts)))

(defmethod builder :center-loss-output-layer [opts]
  (any-layer-builder (CenterLossOutputLayer$Builder.) (:center-loss-output-layer opts)))

(defmethod builder :convolution-1d-layer [opts]
  (any-layer-builder (Convolution1DLayer$Builder.) (:convolution-1d-layer opts)))

(defmethod builder :dropout-layer [opts]
  (any-layer-builder (DropoutLayer$Builder.) (:dropout-layer opts)))

(defmethod builder :global-pooling-layer [opts]
  (any-layer-builder (GlobalPoolingLayer$Builder.) (:global-pooling-layer opts)))

(defmethod builder :subsampling-1d-layer [opts]
  (any-layer-builder (Subsampling1DLayer$Builder.) (:subsampling-1d-layer opts)))

(defmethod builder :zero-padding-layer [opts]
  (let [data (:zero-padding-layer opts)
        {:keys [pad-top pad-bot pad-left pad-right
                pad-height pad-width padding]} data]
    (cond
      (u/contains-many? data :pad-top :pad-bot :pad-left :pad-right)
      (any-layer-builder (ZeroPaddingLayer$Builder. pad-top pad-bot pad-left pad-right)
                         (:zero-padding-layer opts))
      (u/contains-many? data :pad-height :pad-width)
      (any-layer-builder (ZeroPaddingLayer$Builder. pad-height pad-width)
                         (:zero-padding-layer opts))
      :else
      (any-layer-builder (ZeroPaddingLayer$Builder. padding)
                         (:zero-padding-layer opts)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns based on multimethod for documentation purposes
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; write specs to check for correct keys given a layer type
(defn activation-layer-builder
  "creates an activation layer with params supplied in opts map.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in and :n-out to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer"
  [{:keys [activation-fn adam-mean-decay adam-var-decay
           bias-init bias-learning-rate dist drop-out epsilon
           gradient-normalization gradient-normalization-threshold
           l1 l2 layer-name learning-rate learning-rate-policy
           learning-rate-schedule momentum momentum-after rho
           rms-decay updater weight-init n-in n-out l1-bias l2-bias]
    :or {}
    :as opts}]
  (builder {:activation-layer opts}))

(defn output-layer-builder
  "creates an output layer with params supplied in opts map.

  Output layer with different objective co-occurrences for different objectives.
  This includes classification as well as regression

  base case opts can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :loss-fn

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :loss-fn (keyword) Error measurement at the output layer
   opts are: :mse, :expll :xent :mcxent :rmse-xent :squared-loss
            :negativeloglikelihood"

  [{:keys [activation-fn adam-mean-decay adam-var-decay
           bias-init bias-learning-rate dist drop-out epsilon
           gradient-normalization gradient-normalization-threshold
           l1 l2 layer-name learning-rate learning-rate-policy
           learning-rate-schedule momentum momentum-after rho
           rms-decay updater weight-init n-in n-out loss-fn
           l1-bias l2-bias]
    :or {}
    :as opts}]
  (builder {:output-layer opts}))

(defn rnn-output-layer-builder
  "creates a rnn output layer with params supplied in opts map.

  base case opts can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :loss-fn

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :loss-fn (keyword) Error measurement at the output layer
   opts are: :mse, :expll :xent :mcxent :rmse-xent :squared-loss
            :negativeloglikelihood"

  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out loss-fn l1-bias l2-bias]
    :or {}
    :as opts}]
  (builder {:rnn-output-layer opts}))

(defn auto-encoder-layer-builder
  "creates an autoencoder layer with params supplied in opts map.

  Autoencoder. Add Gaussian noise to input and learn a reconstruction function.

  base case opts can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :loss-fn, :corruption-level :sparsity

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :pre-train-iterations (int)

  :visible-bias-init (double)

  :loss-fn (keyword) Error measurement at the output layer
   opts are: :mse, :expll :xent :mcxent :rmse-xent :squared-loss
            :negativeloglikelihood

  :corruption-level (double) turns the autoencoder into a denoising autoencoder:
   see http://deeplearning.net/tutorial/dA.html (code examples in python) and
   http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/217

  :sparsity (double), see http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity

   The denoising auto-encoder is a stochastic version of the auto-encoder. Intuitively,
   a denoising auto-encoder does two things: try to encode the input (preserve the information about the input),
   and try to undo the effect of a corruption process stochastically applied to the input of the auto-encoder.
   The latter can only be done by capturing the statistical dependencies between the inputs."

  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out loss-fn corruption-level
           sparsity l1-bias l2-bias pre-train-iterations visible-bias-init]
    :or {}
    :as opts}]
  (builder {:auto-encoder opts}))

(defn rbm-layer-builder
  "creates a rbm layer with params supplied in opts map.

  Restricted Boltzmann Machine. Markov chain with gibbs sampling.
  Supports the following visible units: BINARY GAUSSIAN SOFTMAX LINEAR
  Supports the following hidden units: RECTIFIED BINARY GAUSSIAN SOFTMAX
  Based on Hinton et al.'s work Great reference:
  http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/239

  base case opts can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :loss-fn, :hidden-unit, :visible-unit, :k, :sparsity

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :pre-train-iterations (int)

  :visible-bias-init (double)

  :loss-fn (keyword) Error measurement at the output layer
   opts are: :mse, :expll :xent :mcxent :rmse-xent :squared-loss
            :negativeloglikelihood

  :hidden-unit (keyword), see https://deeplearning4j.org/restrictedboltzmannmachine
   keyword is one of: :softmax, :binary, :gaussian, :identity, :rectified

  :visible-unit (keyword), see above (hidden-unit link)
   keyword is one of: :softmax, :binary, :gaussian, :identity, :linear

  :k (int), the number of times you run contrastive divergence (gradient calc)
  see https://deeplearning4j.org/glossary.html#contrastivedivergence

  :sparsity (double), see http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity"

  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out loss-fn hidden-unit visible-unit
           sparsity l1-bias l2-bias pre-train-iterations visible-bias-init]
    :or {}
    :as opts}]
  (builder {:rbm opts}))

(defn graves-bidirectional-lstm-layer-builder
  "creates a graves-bidirectional-lstm layer with params supplied in opts map.

  LSTM recurrent net, based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
   http://www.cs.toronto.edu/~graves/phd.pdf

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :forget-gate-bias-init to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :forget-gate-bias-init (double), sets the forget gate bias initializations for LSTM

  :gate-activation-fn (keyword) activation-fn for the gate in an LSTM neuron.
   -can take on the same values as activation-fn"
  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out forget-gate-bias-init
           l1-bias l2-bias gate-activation-fn]
    :or {}
    :as opts}]
  (builder {:graves-bidirectional-lstm opts}))

(defn garves-lstm-layer-builder
  "creates a graves-lstm layer with params supplied in opts map.

  LSTM recurrent net, based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
  http://www.cs.toronto.edu/~graves/phd.pdf

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :forget-gate-bias-init to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :forget-gate-bias-init (double), sets the forget gate bias initializations for LSTM

  :gate-activation-fn (keyword) activation-fn for the gate in an LSTM neuron.
   -can take on the same values as activation-fn"
  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out forget-gate-bias-init
           l1-bias l2-bias gate-activation-fn]
    :or {}
    :as opts}]
  (builder {:graves-lstm opts}))

(defn batch-normalization-layer-builder
  "creates a batch-normalization layer with params supplied in opts map.

  Batch normalization configuration

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :beta, :decay, :eps, :gamma,
                    :is-mini-batch, :lock-gamma-beta to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :beta (double), only used when :lock-gamma-beta is true, sets beta, defaults to 0.0

  :decay (double), Decay value to use for global stats calculation (estimation of mean and variance)

  :eps (double), Epsilon value for batch normalization; small floating point value added to variance
   Default: 1e-5

  :gamma (double), only used when :lock-gamma-beta is true, sets gamma, defaults to 1.0

  :is-mini-batch (boolean), If doing minibatch training or not. Default: true.
   Under most circumstances, this should be set to true.
   Affects how globabl mean/variance estimates are calc'd

  :lock-gamma-beta (boolean), true: lock the gamma and beta parameters to the values for each activation,
   specified by :gamma (double) and :beta (double).
   Default: false -> learn gamma and beta parameter values during network training."

  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out beta decay eps gamma
           is-mini-batch lock-gamma-beta l1-bias l2-bias]
    :or {}
    :as opts}]
  (builder {:batch-normalization opts}))

(defn convolutional-layer-builder
  "creates a convolutional layer with params supplied in opts map.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :convolution-type, :cudnn-algo-mode,
                    :kernel-size, :padding, :stride  to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :convolution-mode (keyword), one of :strict, :same, :truncate
   -see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ConvolutionMode.html

  :cudnn-algo-mode (keyword), either :no-workspace or :prefer-fastest
   Default: :prefer-fastest but :no-workspace uses less memory

  :kernel-size (int), Size of the convolution rows/columns (height and width of the kernel)

  :padding (int), allow us to control the spatial size of the output volumes,
    pad the input volume with zeros around the border.

  :stride (int), filter movement speed across pixels.
   see http://cs231n.github.io/convolutional-networks/"

  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out convolution-mode
           cudnn-algo-mode kernel-size padding stride l1-bias l2-bias]
    :or {}
    :as opts}]
  (builder (:convolutional-layer opts)))

(defn dense-layer-builder
  "creates a dense layer with params supplied in opts map.

  Dense layer: fully connected feed forward layer trainable by backprop.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in and :n-out to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer"

  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out l1-bias l2-bias]
    :or {}
    :as opts}]
  (builder {:dense-layer opts}))

(defn embedding-layer-builder
  "creates an embedding layer with params supplied in opts map.

  feed-forward layer that expects single integers per example as input
  (class numbers, in range 0 to numClass-1) as input. This input has shape [numExamples,1]
  instead of [numExamples,numClasses] for the equivalent one-hot representation.

  Mathematically, EmbeddingLayer is equivalent to using a DenseLayer with a one-hot
  representation for the input; however, it can be much more efficient with a large
  number of classes (as a dense layer + one-hot input does a matrix multiply with all but one value being zero).
   -can only be used as the first layer for a network
   -For a given example index i, the output is activationFunction(weights.getRow(i) + bias),
    hence the weight rows can be considered a vector/embedding for each example.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in and :n-out to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer"
  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out l1-bias l2-bias]
    :or {}
    :as opts}]
  (builder (:embedding-layer opts)))

(defn local-response-normalization-layer-builder
  "creates a local-response-normalization layer with params supplied in opts map.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :alpa, :beta, :k, :n to the param map.

  :alpha (double) LRN scaling constant alpha. Default: 1e-4

  :beta (double) Scaling constant beta. Default: 0.75

  :k (double) LRN scaling constant k. Default: 2

  :n (double) Number of adjacent kernel maps to use when doing LRN. default: 5"

  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init alpha beta k n l1-bias l2-bias]
    :or {}
    :as opts}]
  (builder {:local-response-normalization opts}))

(defn subsampling-layer-builder
  "creates a subsampling layer with params supplied in opts map.

  Subsampling layer also referred to as pooling in convolution neural nets.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :kernel-size, :padding, :pooling-type, :stride to the param map.

  :convolution-mode (keyword), one of :strict, :same, :truncate
   -see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ConvolutionMode.html

  :eps (double), Epsilon value for batch normalization; small floating point value added to variance
   Default: 1e-5

  :pnorm (int) P-norm constant
  -Only used if using PoolingType.PNORM for the pooling type

  :kernel-size :kernel-size (int), Size of the convolution rows/columns (height and width of the kernel)

  :padding (int) padding in the height and width dimensions

  :pooling-type (keyword) progressively reduces the spatial size of the representation to reduce
   the amount of features and the computational complexity of the network.
  one of: :avg, :max, :sum, :pnorm, :none

  :stride (int), filter movement speed across pixels.
   see http://cs231n.github.io/convolutional-networks/"

  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init kernel-size padding pooling-type
           stride l1-bias l2-bias convolution-mode eps pnorm]
    :or {}
    :as opts}]
  (builder {:subsampling-layer opts}))

(defn variational-auto-encoder-builder
  ;; add in this documentation
  ""
  [{:keys [decoder-layer-sizes encoder-layer-sizes loss-fn
           n-out num-samples pzx-activation-function reconstruction-dist
           n-in pre-train-iterations visible-bias-init activation-fn
           adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 l1-bias l2-bias
           learning-rate learning-rate-policy learning-rate-schedule momentum momentum-after
           layer-name rho rms-decay updater weight-init]
    :or {}
    :as opts}]
  (builder {:variational-auto-encoder opts}))


(defn loss-layer-builder
  "creates a loss-layer with params supplied in opts map.

  LossLayer is a flexible output layer that performs a loss function on an input without MLP logic.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :loss-fn, :n-in, :n-out

  :loss-fn (keyword) Error measurement at the output layer
   opts are: :mse, :expll :xent :mcxent :rmse-xent :squared-loss
            :negativeloglikelihood"
  [{:keys [loss-fn n-in n-out activation-fn adam-mean-decay adam-var-decay
           bias-init bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l1-bias l2 l2-bias layer-name
           learning-rate learning-rate-policy learning-rate-schedule momentum
           momentum-after rho rms-decay updater weight-init]
    :or {}
    :as opts}]
  (builder {:loss-layer opts}))

(defn center-loss-output-layer-builder
  "creates a center-loss-output layer with params supplied in opts map.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  Center loss is similar to triplet loss except that it enforces intraclass consistency
  and doesn't require feed forward of multiple examples.
  Center loss typically converges faster for training ImageNet-based convolutional networks.
  If example x is in class Y, ensure that embedding(x) is close to average(embedding(y)) for all examples y in Y

  this builder adds :alpha, :gradient-check, :lambda

  :alpha (double)

  :gradient-check (boolean)

  :lambda (double)"
  [{:keys [loss-fn n-in n-out activation-fn adam-mean-decay adam-var-decay
           bias-init bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l1-bias l2 l2-bias layer-name
           learning-rate learning-rate-policy learning-rate-schedule momentum
           momentum-after rho rms-decay updater weight-init alpha gradient-check
           lambda]
    :or {}
    :as opts}]
  (builder {:center-loss-output-layer opts}))

(defn convolution-1d-layer-builder
  "creates a convolutional layer with params supplied in opts map.

  1D (temporal) convolutional layer. Currently, we just subclass off the ConvolutionLayer
  and hard code the width dimension to 1. Also, this layer accepts RNN InputTypes
  instead of CNN InputTypes. This approach treats a multivariate time series with L
  timesteps and P variables as an L x 1 x P image (L rows high, 1 column wide, P channels deep).
  The kernel should be H

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :convolution-type, :cudnn-algo-mode,
                    :kernel-size, :padding, :stride  to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :convolution-mode (keyword), one of :strict, :same, :truncate
   -see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ConvolutionMode.html

  :cudnn-algo-mode (keyword), either :no-workspace or :prefer-fastest
   Default: :prefer-fastest but :no-workspace uses less memory

  :kernel-size (int), Size of the convolution rows/columns (height and width of the kernel)

  :padding (int), allow us to control the spatial size of the output volumes,
    pad the input volume with zeros around the border.

  :stride (int), filter movement speed across pixels.
   see http://cs231n.github.io/convolutional-networks/"
  [{:keys [convolution-mode cudnn-algo-mode kernel-size padding stride
           n-in n-out activation-fn adam-mean-decay adam-var-decay
           bias-init bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l1-bias l2 l2-bias layer-name
           learning-rate learning-rate-policy learning-rate-schedule momentum
           momentum-after rho rms-decay updater weight-init]
    :or {}
    :as opts}]
  (builder {:convolution-1d-layer opts}))

(defn dropout-layer-builder
  "creates a drop-out layer with params supplied in opts map.

  see any-layer-builder for param descriptions"
  [{:keys [n-in n-out activation-fn adam-mean-decay adam-var-decay
           bias-init bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l1-bias l2 l2-bias layer-name
           learning-rate learning-rate-policy learning-rate-schedule momentum
           momentum-after rho rms-decay updater weight-init]
    :or {}
    :as opts}]
  (builder {:dropout-layer opts}))

(defn global-pooling-layer-builder
  "creates a global pooling layer with params supplied in opts map.

  Global pooling layer - used to do pooling over time for RNNs, and 2d pooling for CNNs.

  Global pooling layer can also handle mask arrays when dealing with variable length inputs.
  -Mask arrays are assumed to be 2d, and are fed forward through the network during training or post-training forward pass:
   -Time series: mask arrays are shape [minibatchSize, maxTimeSeriesLength] and contain values 0 or 1 only
   -CNNs: mask have shape [minibatchSize, height] or [minibatchSize, width].
    - Important: the current implementation assumes that for CNNs + variable length (masking),
      the input shape is [minibatchSize, depth, height, 1] or [minibatchSize, depth, 1, width]
       respectively. This is the case with global pooling in architectures like CNN for sentence classification.

  Behaviour with default settings:
   -3d (time series) input with shape [minibatchSize, vectorSize, timeSeriesLength] -> 2d output [minibatchSize, vectorSize]
   -4d (CNN) input with shape [minibatchSize, depth, height, width] -> 2d output [minibatchSize, depth]

  Alternatively, by setting collapseDimensions = false in the configuration,
  it is possible to retain the reduced dimensions as 1s:
   -this gives [minibatchSize, vectorSize, 1] for RNN output
    and [minibatchSize, depth, 1, 1] for CNN output.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  adds :collapse-dimensions, :pnorm, :pooling-dimensions, :pooling-type

  :collapse-dimensions (boolean) Whether to collapse dimensions when pooling or not.
   -Usually you *do* want to do this. Default: true.
    -If true:
      -3d (time series) input with shape [minibatchSize, vectorSize, timeSeriesLength] -> 2d output [minibatchSize, vectorSize]
      -4d (CNN) input with shape [minibatchSize, depth, height, width] -> 2d output [minibatchSize, depth]
    -If false:
      -3d (time series) input with shape [minibatchSize, vectorSize, timeSeriesLength] -> 3d output [minibatchSize, vectorSize, 1]
      -4d (CNN) input with shape [minibatchSize, depth, height, width] -> 2d output [minibatchSize, depth, 1, 1]

  :pnorm (int) P-norm constant
  -Only used if using PoolingType.PNORM for the pooling type

  :pooling-dimensions (int) Pooling dimensions
  -Note: most of the time, this doesn't need to be set, and the defaults can be used.
   -Default for RNN data: pooling dimension 2 (time).
   -Default for CNN data: pooling dimensions 2,3 (height and width)

  :pooling-type (keyword) progressively reduces the spatial size of the representation to reduce
    the amount of features and the computational complexity of the network.
    one of: :avg, :max, :sum, :pnorm, :none"
  [{:keys [activation-fn adam-mean-decay adam-var-decay
           bias-init bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l1-bias l2 l2-bias layer-name
           learning-rate learning-rate-policy learning-rate-schedule momentum
           momentum-after rho rms-decay updater weight-init collapse-dimensions
           pnorm pooling-dimensions pooling-type]
    :or {}
    :as opts}]
  (builder {:global-pooling-layer opts}))

(defn subsampling-1d-layer-builder
  "creates a 1d subsampling layer with the params supplied in opts map.

   1D (temporal) subsampling layer. Currently, we just subclass off the SubsamplingLayer and hard
   code the width dimension to 1. Also, this layer accepts RNN InputTypes instead of CNN InputTypes.

   This approach treats a multivariate time series with L timesteps and P
    variables as an L x 1 x P image (L rows high, 1 column wide, P channels deep).

   The kernel should be H

  adds :convolution-mode, :eps, :kernel-size, :padding, :pnorm, :pooling-type, :stride

  :convolution-mode (keyword), one of :strict, :same, :truncate
   -see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ConvolutionMode.html

  :eps (double), Epsilon value for batch normalization; small floating point value added to variance
   Default: 1e-5

  :kernel-size (int), Size of the convolution rows/columns (height and width of the kernel)

  :padding (int), allow us to control the spatial size of the output volumes,
    pad the input volume with zeros around the border.

  :pnorm (int) P-norm constant
  -Only used if using PoolingType.PNORM for the pooling type

  :pooling-type (keyword) progressively reduces the spatial size of the representation to reduce
    the amount of features and the computational complexity of the network.
    one of: :avg, :max, :sum, :pnorm, :none

  :stride (int), filter movement speed across pixels.
   see http://cs231n.github.io/convolutional-networks/"
  [{:keys [activation-fn adam-mean-decay adam-var-decay
           bias-init bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l1-bias l2 l2-bias layer-name
           learning-rate learning-rate-policy learning-rate-schedule momentum
           momentum-after rho rms-decay updater weight-init convolution-mode
           eps kernel-size padding pnorm pooling-type stride]
    :or {}
    :as opts}]
  (builder {:subsampling-1d-layer opts}))

(defn zero-padding-layer-builder
  "builds a zero-padding layer with the params supplied in opts map.

  Zero padding layer for convolutional neural networks.
  -Allows padding to be done separately for top/bottom/left/right

  adds :padding, :pad-height, :pad-width, :pad-top :pad-bot, :pad-left, :pad-right

  :padding (int), allow us to control the spatial size of the output volumes,
    pad the input volume with zeros around the border.

  :pad-height (int)

  :pad-width (int)

  :pad-top (int)

  :pad-bot (int)

  :pad-left (int)

  :pad-right (int)"
  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l1-bias l2 l2-bias layer-name
           learning-rate learning-rate-policy learning-rate-schedule momentum
           momentum-after rho rms-decay updater weight-init padding pad-height
           pad-width pad-top pad-bot pad-left pad-right]
    :or {}
    :as opts}]
  (builder {:zero-padding-layer opts}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; examples
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(comment
 (builder {:graves-lstm {:activation-fn :softmax
                            :adam-mean-decay 0.3
                            :adam-var-decay 0.5
                            :bias-init 0.3
                              :dist {:binomial {:number-of-trials 0, :probability-of-success 0.08}}
                            :drop-out 0.01
                            :gradient-normalization :clip-l2-per-layer
                            :gradient-normalization-threshold 0.1
                            :l1 0.02
                            :l2 0.002
                            :learning-rate 0.95
                            :learning-rate-after {1000 0.5}
                            :learning-rate-score-based-decay-rate 0.001
                            :momentum 0.9
                            :momentum-after {10000 1.5}
                            :layer-name "test"
                            :rho 0.5
                            :rms-decay 0.01
                            :updater :adam
                           ;; :weight-init :normalized
                            :n-in 30
                            :n-out 30
                            :forget-gate-bias-init 0.12}})

 (type (garves-lstm-layer-builder {:activation-fn :softmax
                             :adam-mean-decay 0.3
                             :adam-var-decay 0.5
                             :bias-init 0.3
                             :dist {:binomial {:number-of-trials 0, :probability-of-success 0.08}}
                             :drop-out 0.01
                             :gradient-normalization :clip-l2-per-layer
                             :gradient-normalization-threshold 0.1
                             :l1 0.02
                             :l2 0.002
                             :learning-rate 0.95
                             :learning-rate-after {1000 0.5}
                             :learning-rate-score-based-decay-rate 0.001
                             :momentum 0.9
                             :momentum-after {10000 1.5}
                             :layer-name "test"
                             :rho 0.5
                             :rms-decay 0.01
                             :updater :adam
                             ;;:weight-init :normalized
                             :n-in 30
                             :n-out 30
                             :forget-gate-bias-init 0.12}))
)
