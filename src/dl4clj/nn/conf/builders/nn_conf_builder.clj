(ns dl4clj.nn.conf.builders.nn-conf-builder
  (:require [dl4clj.nn.conf.distribution.distribution :as distribution]
            [dl4clj.nn.conf.builders.builders :as layer-builders]
            [dl4clj.nn.conf.builders.multi-layer-builders :as multi-layer]
            [dl4clj.nn.conf.step-fns :as step-functions]
            [dl4clj.nn.conf.constants :as constants])
  (:import [org.deeplearning4j.nn.conf
            NeuralNetConfiguration$Builder
            NeuralNetConfiguration$ListBuilder
            ComputationGraphConfiguration$GraphBuilder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]))

(defn nn-conf-builder
  "creates a nn-conf based on the supplied args.  There are two types of args here:
   1) network args
      - these are params that affect the network as a whole instead of just at the layer level

   2) global layer args
      - these params will not overwrite those set at the layer level
      - will replace layer params that were not set at the layer level
      - args from layer.builder, the base class for all layers

  If you only supply one or no layer, will be a neuralnetconfiguration
  if you supply layers, will be a multilayerconfiguration

  Params for the neural network are:

  :nn-builder (nn-conf-builder), a preexisting nn-conf-builder.  If not supplied,
   a fresh builder will be used

  :iterations (int) Number of optimization iterations

  :lr-policy-decay-rate (double) Sets the decay rate for the learning rate decay policy.

  :lr-policy-power (double) Sets the power used for learning rate inverse policy

  :lr-policy-steps (double) Sets the number of steps used for learning decay rate steps policy

  :max-num-line-search-iterations (int) Maximum number of line search iterations

  :mini-batch? (boolean) Process input as minibatch instead of as a full dataset

  :minimize? (boolean) Objective function to minimize or maximize cost function
   default is true

  :use-drop-connect? (boolean) Multiply the weight by a binomial sampling wrt the dropout probability.
   Dropconnect probability is set using :drop-out(double); this is the probability of retaining a weight

  :optimization-algo (keyword) Optimization algorithm to use
   one of: :line-gradient-descent, :conjugate-gradient, :hessian-free (deprecated),
           :lbfgs, :stochastic-gradient-descent

  :lr-score-based-decay-rate (double)
   Rate to decrease :learning-rate by when the score stops improving

  :regularization? (boolean) Whether or not to use regularization

  :seed (int or long) Random number generator seed, Used for reproducability between runs

  :step-fn (keyword) Step functions to apply for back track line search
   Only applies for line search optimizers: Line Search SGD, Conjugate Gradient, LBFGS
   one of: :default-step-fn (default), :negative-default-step-fn :gradient-step-fn (for SGD),
   :negative-gradient-step-fn
   - can also be the object created by one of the new-fns
   - ie. (new-default-step-fn)

  :convolution-mode (keyword), one of: :strict, :truncate, :same
   - see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ConvolutionMode.html

  :build? (boolean), defaults to false, builds the nn-conf
   - defaults to false because the resulting list builder is needed when passing
     the result of this fn to multi-layer-config-builder.  If there are no additional
     params you want to added the configuration (from multi-layer-config-builder) then
     set built? to true

  :layer {:layer-type {config-opts vals}} map of layer config params.
    see dl4clj.nn.conf.builders.builders (can also be a fn used to create a layer)
    :layer will be ignored if :layers is supplied, layer only creates a single layer

  :layers {idx (int) {:layer-type {config-opts vals}}}
   see dl4clj.nn.conf.builders.builders

  Params for layers are:

  :default-activation-fn (keyword) default global activation fn,
   wont overwrite activation functions specificed in the layer or layers map.
   opts are :cube, :elu, :hard-sigmoid, :hard-tanh, :identity, :leaky-relu
            :relu, :r-relu, :sigmoid, :soft-max, :soft-plus, :soft-sign, :tanh
            :rational-tanh
    - does replace non-specified activation fns in layers

  :default-adam-mean-decay (double) Mean decay rate for Adam updater

  :default-adam-var-decay (double) Variance decay rate for Adam updater

  :default-bias-init (double) Constant for bias initialization

  :default-bias-learning-rate (double) Bias learning rate

  :default-dist (map) distribution to sample initial weights from, one of:
        binomial-distribution {:binomial {:number-of-trails int :probability-of-success double}}
        normal-distribution {:normal {:mean double :std double}}
        uniform-distribution {:uniform {:lower double :upper double}}
   - can also use one of the creation fns
   - ie. (new-normal-distribution :mean 0 :std 1)

  :default-drop-out (double) Dropout probability

  :default-epsilon (double) Epsilon value for updaters: Adagrad and Adadelta

  :default-gradient-normalization (keyword) gradient normalization strategy,
   These are applied on raw gradients, before the gradients are passed to the updater
   (SGD, RMSProp, Momentum, etc)
   one of: :none (default), :renormalize-l2-per-layer, :renormalize-l2-per-param-type,
           :clip-element-wise-absolute-value, :clip-l2-per-layer, :clip-l2-per-param-type
   reference: https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/GradientNormalization.html

  :default-gradient-normalization-threshold (double) Threshold for gradient normalization,
   only used for :clip-l2-per-layer, :clip-l2-per-param-type, :clip-element-wise-absolute-value,
   L2 threshold for first two types of clipping or absolute value threshold for the last type

  :default-l1 (double) L1 regularization coefficient

  :default-l1-bias (double), L1 regularization coef for the bias

  :default-l2 (double) L2 regularization coefficient used when regularization is set to true

  :default-l2-bias (double) L2 regularization coef for the bias

  :default-learning-rate (double) Paramter that controls the learning rate

  :default-learning-rate-policy (keyword) How to decay learning rate during training
   see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/LearningRatePolicy.html
   one of :none, :exponential, :inverse, :poly, :sigmoid, :step, :torch-step :schedule :score

  :default-learning-rate-schedule {int double} Map of the iteration to the learning rate to apply at that iteration

  :default-momentum (double) Momentum rate used only when the :updater is set to :nesterovs

  :default-momentum-after {int double} Map of the iteration to the momentum rate to apply at that iteration
   also only used when :updater is :nesterovs

  :default-rho (double) Ada delta coefficient

  :default-rms-decay (double) Decay rate for RMSProp, only applies if using :updater :RMSPROP

  :default-updater (keyword) Gradient updater,
   one of: :adagrad, :sgd, :adam, :adadelta, :nesterovs, :adagrad, :rmsprop, :none, :custom

  :default-weight-init (keyword) Weight initialization scheme
  see https://deeplearning4j.org/doc/org/deeplearning4j/nn/weights/WeightInit.html (use cases)
  one of: :distribution, :zero, :sigmoid-uniform, :uniform, :xavier, :xavier-uniform
          :xavier-fan-in, :xavier-legacy, :relu, :relu-uniform"
  [& {:keys [default-activation-fn default-adam-mean-decay default-adam-var-decay
             default-bias-init default-bias-learning-rate default-dist default-drop-out
             default-epsilon default-gradient-normalization default-gradient-normalization-threshold
             default-l1 default-l2 default-l1-bias default-l2-bias layer layers
             default-learning-rate default-learning-rate-policy default-learning-rate-schedule
             default-momentum default-momentum-after default-rho default-rms-decay
             default-updater default-weight-init nn-builder
             ;; nn conf args
             iterations lr-policy-decay-rate lr-policy-power
             lr-policy-steps max-num-line-search-iterations mini-batch? minimize?
             use-drop-connect? optimization-algo lr-score-based-decay-rate
             regularization? seed step-fn convolution-mode build?]
      :or {build? false
           nn-builder (NeuralNetConfiguration$Builder.)}
      :as opts}]
  (cond-> nn-builder
    (contains? opts :default-activation-fn) (.activation (constants/value-of
                                                         {:activation-fn
                                                          default-activation-fn}))
    (contains? opts :default-adam-mean-decay) (.adamMeanDecay default-adam-mean-decay)
    (contains? opts :default-adam-var-decay) (.adamVarDecay default-adam-var-decay)
    (contains? opts :convolution-mode) (.convolutionMode (constants/value-of
                                                          {:convolution-mode
                                                           convolution-mode}))
    (contains? opts :default-bias-init) (.biasInit default-bias-init)
    (contains? opts :default-bias-learning-rate) (.biasLearningRate default-bias-learning-rate)
    (contains? opts :default-dist) (.dist (if (map? default-dist) (distribution/distribution default-dist)
                                              default-dist))
    (contains? opts :default-drop-out) (.dropOut default-drop-out)
    (contains? opts :default-epsilon) (.epsilon default-epsilon)
    (contains? opts :default-gradient-normalization) (.gradientNormalization
                                                      (constants/value-of
                                                       {:gradient-normalization
                                                        default-gradient-normalization}))
    (contains? opts :default-gradient-normalization-threshold) (.gradientNormalizationThreshold
                                                                default-gradient-normalization-threshold)
    (contains? opts :iterations) (.iterations iterations)
    (contains? opts :default-l1) (.l1 default-l1)
    (contains? opts :default-l1-bias) (.l1Bias default-l1-bias)
    (contains? opts :default-l2) (.l2 default-l2)
    (contains? opts :default-l2-bias) (.l2Bias default-l2-bias)
    (contains? opts :default-learning-rate) (.learningRate default-learning-rate)
    (contains? opts :default-learning-rate-policy) (.learningRateDecayPolicy
                                            (constants/value-of
                                             {:learning-rate-policy
                                              default-learning-rate-policy}))
    (contains? opts :default-learning-rate-schedule) (.learningRateSchedule
                                                      default-learning-rate-schedule)
    (contains? opts :lr-score-based-decay-rate) (.learningRateScoreBasedDecayRate
                                                 lr-score-based-decay-rate)
    (contains? opts :lr-policy-decay-rate) (.lrPolicyDecayRate lr-policy-decay-rate)
    (contains? opts :lr-policy-power) (.lrPolicyPower lr-policy-power)
    (contains? opts :lr-policy-steps) (.lrPolicySteps lr-policy-steps)
    (contains? opts :max-num-line-search-iterations) (.maxNumLineSearchIterations
                                                      max-num-line-search-iterations)
    (contains? opts :mini-batch?) (.miniBatch mini-batch?)
    (contains? opts :minimize?) (.minimize minimize?)
    (contains? opts :default-momentum) (.momentum default-momentum)
    (contains? opts :default-momentum-after) (.momentumAfter default-momentum-after)
    (contains? opts :optimization-algo) (.optimizationAlgo
                                         (constants/value-of
                                          {:optimization-algorithm
                                           optimization-algo}))
    (contains? opts :regularization?) (.regularization regularization?)
    (contains? opts :default-rho) (.rho default-rho)
    (contains? opts :default-rms-decay) (.rmsDecay default-rms-decay)
    (contains? opts :seed) (.seed seed)
    (contains? opts :step-fn) (.stepFunction (if (keyword? step-fn)
                                               (step-functions/step-fn step-fn)
                                               step-fn))
    (contains? opts :default-updater) (.updater (constants/value-of {:updater default-updater}))
    (contains? opts :use-drop-connect?) (.useDropConnect use-drop-connect?)
    (contains? opts :default-weight-init) (.weightInit (constants/value-of
                                                        {:weight-init default-weight-init}))
    (and (contains? opts :layer) (map? layer)) (.layer (layer-builders/builder layer))
    (and (contains? opts :layer) (false? (map? layer))) (.layer layer)
    (contains? opts :layers) (multi-layer/list-builder layers)
    (true? build?) .build))
