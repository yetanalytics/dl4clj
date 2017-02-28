(ns ^{:doc "see https://deeplearning4j.org/glossary for param descriptions"}
    dl4clj.nn.conf.builders.builders
  (:require [dl4clj.nn.conf.distribution.distribution :as distribution]
            [dl4clj.nn.conf.constants :as constants])
  (:import
   [org.deeplearning4j.nn.conf.layers ActivationLayer$Builder
    OutputLayer$Builder RnnOutputLayer$Builder AutoEncoder$Builder
    RBM$Builder GravesBidirectionalLSTM$Builder GravesLSTM$Builder
    BatchNormalization$Builder ConvolutionLayer$Builder DenseLayer$Builder
    EmbeddingLayer$Builder LocalResponseNormalization$Builder SubsamplingLayer$Builder]))

(defn layer-type
  "dispatch fn for builder"
  [opts]
  (first (keys opts)))

(defmulti builder
  "multimethod that builds a layer based on the supplied type and opts"
  layer-type)

(defn any-layer-builder
  "creates any type of layer given a builder and param map

  params shared between all layers:

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

  :gradient-normalization-threshold (double) Threshold for gradient normalization,
   only used for :clip-l2-per-layer, :clip-l2-per-param-type, :clip-element-wise-absolute-value,
   L2 threshold for first two types of clipping or absolute value threshold for the last type

  :l1 (double) L1 regularization coefficient

  :l2 (double) L2 regularization coefficient used when regularization is set to true

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
   one of: :adagrad, :sgd, :adam, :adadelta, :nesterovs, :adagrad, :rmsprop, :none, :custom

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
                        kernel-size stride padding convolution-type cudnn-algo-mode
                        alpha n pooling-type]
                 :or {}
                 :as opts}]
  (if (contains? opts :activation-fn)
    (.activation
     builder-type (constants/value-of {:activation-fn activation-fn}))
    builder-type)
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
     builder-type (constants/value-of {:gradient-normalization gradient-normalization}))
    builder-type)
  (if (contains? opts :gradient-normalization-threshold)
    (.gradientNormalizationThreshold builder-type gradient-normalization-threshold)
    builder-type)
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
     (constants/value-of {:learning-rate-policy learning-rate-policy}))
    builder-type)
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
    (.updater builder-type (constants/value-of {:updater updater}))
    builder-type)
  (if (contains? opts :weight-init)
    (.weightInit builder-type (constants/value-of {:weight-init weight-init}))
    builder-type)
  (if (contains? opts :n-in)
    (.nIn builder-type n-in) builder-type)
  (if (contains? opts :n-out)
    (.nOut builder-type n-out) builder-type)
  (if (contains? opts :loss-fn)
    (.lossFunction builder-type (constants/value-of {:loss-fn loss-fn}))
    builder-type)
  (if (contains? opts :corruption-level)
    (.corruptionLevel builder-type corruption-level) builder-type)
  (if (contains? opts :sparsity)
    (.sparsity builder-type sparsity) builder-type)
  (if (contains? opts :visible-unit)
    (.visibleUnit builder-type (constants/value-of {:visible-unit visible-unit}))
    builder-type)
  (if (contains? opts :hidden-unit)
    (.hiddenUnit builder-type (constants/value-of {:hidden-unit hidden-unit}))
    builder-type)
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
    (.convolutionType
     builder-type (constants/value-of {:convolution-type convolution-type}))
    builder-type)
  (if (contains? opts :cudnn-algo-mode)
    (.cudnnAlgoMode
     builder-type (constants/value-of {:cudnn-algo-mode cudnn-algo-mode}))
    builder-type)
  (if (contains? opts :alpha)
    (.alpha builder-type alpha) builder-type)
  (if (contains? opts :n)
    (.n builder-type n) builder-type)
  (if (contains? opts :pooling-type)
    (.poolingType builder-type (constants/value-of {:pool-type pooling-type}))
    builder-type)
  (.build builder-type))

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
           rms-decay updater weight-init n-in n-out]
    :or {}
    :as opts}]
  (builder {:activation-layer opts}))

(defn output-layer-builder
  "creates an output layer with params supplied in opts map.

  base case opts can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :loss-fn

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :loss-fn (keyword) Error measurement at the output layer
   opts are: :mse, :l1, :xent, :mcxent, :squared-loss,
             :reconstruction-crossentropy, :negativeloglikelihood,
             :cosine-proximity, :hinge, :squared-hinge, :kl-divergence,
             :mean-absolute-error, :l2, :mean-absolute-percentage-error,
             :mean-squared-logarithmic-error, :poisson"

  [{:keys [activation-fn adam-mean-decay adam-var-decay
           bias-init bias-learning-rate dist drop-out epsilon
           gradient-normalization gradient-normalization-threshold
           l1 l2 layer-name learning-rate learning-rate-policy
           learning-rate-schedule momentum momentum-after rho
           rms-decay updater weight-init n-in n-out loss-fn]
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
   opts are: :mse, :l1, :xent, :mcxent, :squared-loss,
             :reconstruction-crossentropy, :negativeloglikelihood,
             :cosine-proximity, :hinge, :squared-hinge, :kl-divergence,
             :mean-absolute-error, :l2, :mean-absolute-percentage-error,
             :mean-squared-logarithmic-error, :poisson"

  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out loss-fn]
    :or {}
    :as opts}]
  (builder {:rnn-output-layer opts}))

(defn auto-encoder-layer-builder
  "creates an autoencoder layer with params supplied in opts map.

  base case opts can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :loss-fn, :corruption-level :sparsity

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :loss-fn (keyword) Error measurement at the output layer
   opts are: :mse, :l1, :xent, :mcxent, :squared-loss,
             :reconstruction-crossentropy, :negativeloglikelihood,
             :cosine-proximity, :hinge, :squared-hinge, :kl-divergence,
             :mean-absolute-error, :l2, :mean-absolute-percentage-error,
             :mean-squared-logarithmic-error, :poisson

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
           sparsity]
    :or {}
    :as opts}]
  (builder {:auto-encoder opts}))

(defn rbm-layer-builder
  "creates a rbm layer with params supplied in opts map.

  base case opts can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :loss-fn, :hidden-unit, :visible-unit, :k, :sparsity

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :loss-fn (keyword) Error measurement at the output layer
   opts are: :mse, :l1, :xent, :mcxent, :squared-loss,
             :reconstruction-crossentropy, :negativeloglikelihood,
             :cosine-proximity, :hinge, :squared-hinge, :kl-divergence,
             :mean-absolute-error, :l2, :mean-absolute-percentage-error,
             :mean-squared-logarithmic-error, :poisson

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
           sparsity]
    :or {}
    :as opts}]
  (builder {:rbm opts}))

(defn graves-bidirectional-lstm-layer-builder
  "creates a graves-bidirectional-lstm layer with params supplied in opts map.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :forget-gate-bias-init to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :forget-gate-bias-init (double), sets the forget gate bias initializations for LSTM"
  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out forget-gate-bias-init]
    :or {}
    :as opts}]
  (builder {:graves-bidirectional-lstm opts}))

(defn garves-lstm-layer-builder
  "creates a graves-lstm layer with params supplied in opts map.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in, :n-out, :forget-gate-bias-init to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer

  :forget-gate-bias-init (double), sets the forget gate bias initializations for LSTM"
  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out forget-gate-bias-init]
    :or {}
    :as opts}]
  (builder {:graves-lstm opts}))

(defn batch-normalization-layer-builder
  "creates a batch-normalization layer with params supplied in opts map.

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
           is-mini-batch lock-gamma-beta]
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

  :convolution-type (keyword), one of :full, :same, :valid

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
           rho rms-decay updater weight-init n-in n-out convolution-type
           cudnn-algo-mode kernel-size padding stride]
    :or {}
    :as opts}]
  (builder (:convolutional-layer opts)))

(defn dense-layer-builder
  "creates a dense layer with params supplied in opts map.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in and :n-out to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer"

  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out]
    :or {}
    :as opts}]
  (builder {:dense-layer opts}))

(defn embedding-layer-builder
  "creates an embedding layer with params supplied in opts map.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :n-in and :n-out to the param map.

  :n-in (int) number of inputs to a given layer

  :n-out (int) number of outputs for the given layer"
  [{:keys [activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold l1 l2 layer-name learning-rate
           learning-rate-policy learning-rate-schedule momentum momentum-after
           rho rms-decay updater weight-init n-in n-out]
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
           rho rms-decay updater weight-init alpha beta k n]
    :or {}
    :as opts}]
  (builder {:local-response-normalization opts}))

(defn subsampling-layer-builder
  "creates a subsampling layer with params supplied in opts map.

  base case opts descriptions can be found in the doc string of any-layer-builder.

  this builder adds :kernel-size, :padding, :pooling-type, :stride to the param map.

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
           rho rms-decay updater weight-init kernel-size padding pooling-type stride]
    :or {}
    :as opts}]
  (builder {:subsampling-layer opts}))





#_(builder {:graves-lstm {:activation-fn :softmax
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
                          :weight-init :normalized
                          :n-in 30
                          :n-out 30
                          :forget-gate-bias-init 0.12}})

#_(garves-lstm-layer-builder {:activation-fn :softmax
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
                            :weight-init :normalized
                            :n-in 30
                            :n-out 30
                            :forget-gate-bias-init 0.12})
