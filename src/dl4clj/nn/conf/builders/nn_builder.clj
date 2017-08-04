(ns dl4clj.nn.builders.nn-builder
  (:require [dl4clj.nn.conf.builders.builders :as layer-builders]
            [dl4clj.helpers :refer [value-of-helper
                                    distribution-helper
                                    step-fn-helper
                                    pre-processor-helper
                                    input-type-helper]]
            [dl4clj.utils :refer [builder-fn replace-map-vals eval-and-build
                                  generic-dispatching-fn]]
            [clojure.core.match :refer [match]])
  (:import [org.deeplearning4j.nn.conf
            NeuralNetConfiguration$Builder
            NeuralNetConfiguration$ListBuilder
            MultiLayerConfiguration$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]))

(def method-map-nn-builder
  {:default-activation-fn                    '.activation
   :default-adam-mean-decay                  '.adamMeanDecay
   :default-adam-var-decay                   '.adamVarDecay
   :convolution-mode                         '.convolutionMode
   :default-bias-init                        '.biasInit
   :default-bias-learning-rate               '.biasLearningRate
   :default-dist                             '.dist
   :default-drop-out                         '.dropOut
   :default-epsilon                          '.epsilon
   :default-gradient-normalization           '.gradientNormalization
   :default-gradient-normalization-threshold '.gradientNormalizationThreshold
   :iterations                               '.iterations
   :default-l1                               '.l1
   :default-l1-bias                          '.l1Bias
   :default-l2                               '.l2
   :default-l2-bias                          '.l2Bias
   :default-learning-rate                    '.learningRate
   :default-learning-rate-policy             '.learningRateDecayPolicy
   :default-learning-rate-schedule           '.learningRateSchedule
   :lr-score-based-decay-rate                '.learningRateScoreBasedDecayRate
   :lr-policy-decay-rate                     '.lrPolicyDecayRate
   :lr-policy-power                          '.lrPolicyPower
   :lr-policy-steps                          '.lrPolicySteps
   :max-num-line-search-iterations           '.maxNumLineSearchIterations
   :mini-batch?                              '.miniBatch
   :minimize?                                '.minimize
   :default-momentum                         '.momentum
   :default-momentum-after                   '.momentumAfter
   :optimization-algo                        '.optimizationAlgo
   :regularization?                          '.regularization
   :default-rho                              '.rho
   :default-rms-decay                        '.rmsDecay
   :seed                                     '.seed
   :step-fn                                  '.stepFunction
   :default-updater                          '.updater
   :use-drop-connect?                        '.useDropConnect
   :default-weight-init                      '.weightInit})

(def multi-layer-methods
  {:backprop?                                '.backprop
   :backprop-type                            '.backpropType
   :input-pre-processors                     '.inputPreProcessors
   :input-type                               '.setInputType
   :pretrain?                                '.pretrain
   :conf                                     '.confs
   :tbptt-back-length                        '.tBPTTBackwardLength
   :tbptt-fwd-length                         '.tBPTTForwardLength})

(defn layer-builder-helper
  "creates the code for creating layers.  Works when there is a single layer or multiple.

  has logic for working with configuration maps and fn calls"
  [nn-conf-b layers]
  (match [layers]
         ;; is it a single fn call to create the layer?
         [(_ :guard seq?)]
         `(.layer ~nn-conf-b (eval-and-build ~layers))
         ;; is it a config map for creating a single layer?
         [(true :<< #(keyword? (generic-dispatching-fn %)))]
         `(.layer ~nn-conf-b (eval-and-build (layer-builders/builder ~layers)))
         ;; is it a config map for creating multiple layers?
         [(true :<< #(integer? (generic-dispatching-fn %)))]
         (builder-fn `(.list ~nn-conf-b) {:add-layers '.layer}
                     {:add-layers
                      (into []
                            (for [each layers
                                  :let [[idx layer] each]]
                              (match [layer]
                                     [(_ :guard map?)]
                                     [idx `(eval-and-build
                                            (layer-builders/builder ~layer))]
                                     :else
                                     [idx `(eval-and-build ~layer)])))})))

(defn multi-layer-builder-helper
  "creates the code for going from a nn-conf to a mln-conf

  or just stays as a nn-conf if the supplied mln-conf-opts* are empty"
  [mln-conf-opts* builder-with-layers layers]
  (if (empty? mln-conf-opts*)
      ;; if we didnt get passed any mln-conf methods, just return the builder with layers added
      builder-with-layers
      (match [layers]
             ;; if we only had one layer, need to use a mln builder to add mln opts
             [(_ :guard seq?)]
             (builder-fn
              `(MultiLayerConfiguration$Builder.)
              multi-layer-methods
              (assoc mln-conf-opts*
                     :conf `(~list (eval-and-build ~builder-with-layers))))
             ;; if we only had one layer, need to use a mln builder to add mln opts
             [(true :<< #(keyword? (generic-dispatching-fn %)))]
             (builder-fn
              `(MultiLayerConfiguration$Builder.)
              multi-layer-methods
              (assoc mln-conf-opts*
                     :conf `(~list (eval-and-build ~builder-with-layers))))
             ;; if we had multiple layers, evaled code will create the multi-layer-conf builder
             [(true :<< #(integer? (generic-dispatching-fn %)))]
             (builder-fn builder-with-layers multi-layer-methods mln-conf-opts*)
             :else
             ;; we were just passed options for setting up a 0 layer nn-conf and options for setting up a mln
             ;; it is assumed that the user will add layers to this mln later
             (builder-fn
              `(MultiLayerConfiguration$Builder.)
              multi-layer-methods
              (assoc mln-conf-opts*
                     :conf `(~list (eval-and-build ~builder-with-layers)))))))

(defn builder
  ;; come back and update this doc string
  ;; changes like eval-and-build?
  ;; might just move this info over to the wiki
  ;; and say for arg descriptions, see the wiki
  ;; because this doc string would otherwise be huge
  "
  1) network args
      - these are params that affect the network as a whole instead of just at the layer level

   2) global layer args
      - these params will not overwrite those set at the layer level
      - will replace layer params that were not set at the layer level
      - args from layer.builder, the base class for all layers

  3) multi layer args

  If you only supply one or no layer, will return a neural net configuration
  if you supply layers, will return a multi layer configuration
   - you should only supply a config for one of these options
     (layer or layers not both)




  Params for the neural network are:

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

  :convolution-mode (keyword), one of: :strict, :truncate, :same
   - see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/ConvolutionMode.html

  :build? (boolean), defaults to false, builds the nn-conf
   - defaults to false because the resulting list builder is needed when passing
     the result of this fn to multi-layer-config-builder.

   - If there are no additional params you want added to the configuration
     (from multi-layer-config-builder) then set built? to true

  :layers {idx (int) {:layer-type {config-opts vals}}}
   see dl4clj.nn.conf.builders.builders and tests for examples
  {:layer-type {config-opts vals}}



  Params for layers are:

  :default-activation-fn (keyword) default global activation fn,
   wont overwrite activation functions specificed in the layer or layers map.
   opts are :cube, :elu, :hard-sigmoid, :hard-tanh, :identity, :leaky-relu
            :relu, :r-relu, :sigmoid, :soft-max, :soft-plus, :soft-sign, :tanh
            :rational-tanh

  :default-adam-mean-decay (double) Mean decay rate for Adam updater

  :default-adam-var-decay (double) Variance decay rate for Adam updater

  :default-bias-init (double) Constant for bias initialization

  :default-bias-learning-rate (double) Bias learning rate

  :default-dist (map) distribution to sample initial weights from, one of:
        binomial-distribution = {:binomial {:number-of-trails int :probability-of-success double}}
        normal-distribution = {:normal {:mean double :std double}}
        uniform-distribution = {:uniform {:lower double :upper double}}

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
          :xavier-fan-in, :xavier-legacy, :relu, :relu-uniform




  params for multi layer networks are:

  :backprop? (boolean) whether to do backprop or not

  :backprop-type (keyword) the type of backprop, one of :standard or :truncated-bptt

  :input-pre-processors {int pre-processor} ie {0 (new-zero-mean-pre-pre-processor)
                                                1 {:unit-variance-processor {}}}
  specifies the processors, these are used at each layer for doing things like
  normalization and shaping of input.
   - the pre-processors should be config maps but can the code for creating the
     java objects.  If its the code for creating the objects, it must be quoted

  :input-type (map), map of params describing the input
   {(keyword) other-opts}, ie. {:convolutional {:input-height 1 ...}}
    - the first key is one of: :convolutional, :convolutional-flat, :feed-forward, :recurrent
    - other opts: for feedforward and recurrent, supply the :size of the layer
                  for convolutional, supply the height width depth of the layer

  :pretrain? (boolean) Whether to do pre train or not

  :tbptt-back-length (int) When doing truncated BPTT: how many steps of backward should we do?
  Only applicable when :backprop-type = :truncated-bptt

  :tbptt-fwd-length (int) When doing truncated BPTT: how many steps of forward pass
  should we do before doing (truncated) backprop? Only applicable when :backprop-type = :truncated-bptt
   - Typically tBPTTForwardLength parameter is same as the tBPTTBackwardLength parameter,
     but may be larger than it in some circumstances (but never smaller)
   - Ideally your training data time series length should be divisible by this"
  [& {:keys [;; default layer options
             default-activation-fn default-adam-mean-decay default-adam-var-decay
             default-bias-init default-bias-learning-rate default-dist default-drop-out
             default-epsilon default-gradient-normalization
             default-gradient-normalization-threshold default-l1 default-l2
             default-l1-bias default-l2-bias default-learning-rate
             default-learning-rate-policy default-learning-rate-schedule
             default-momentum default-momentum-after default-rho default-rms-decay
             default-updater default-weight-init
             ;; nn conf opts
             iterations lr-policy-decay-rate lr-policy-power
             lr-policy-steps max-num-line-search-iterations mini-batch? minimize?
             use-drop-connect? optimization-algo lr-score-based-decay-rate nn-builder
             regularization? seed step-fn convolution-mode layer layers build? eval?
             ;; multi layer opts
             backprop? backprop-type input-pre-processors input-type pretrain?
             tbptt-back-length tbptt-fwd-length

             ;; do we want the code or the evaluated and built code?
             eval-and-build?]
      :or {nn-builder `(NeuralNetConfiguration$Builder.)
           eval-and-build? true}
      :as opts}]
  (let [;; set up code for value of and other objects/enums
        a (if default-activation-fn
           (value-of-helper :activation-fn default-activation-fn))
        c-m (if convolution-mode
             (value-of-helper :convolution-mode convolution-mode))
        d (if default-dist
            (distribution-helper default-dist))
        g-norm (if default-gradient-normalization
                 (value-of-helper :gradient-normalization default-gradient-normalization))
        lr-p (if default-learning-rate-policy
               (value-of-helper :learning-rate-policy default-learning-rate-policy))
        o-a (if optimization-algo
              (value-of-helper :optimization-algorithm optimization-algo))
        s-f (if step-fn
              (step-fn-helper step-fn))
        u (if default-updater
            (value-of-helper :updater default-updater))
        w (if default-weight-init
            (value-of-helper :weight-init default-weight-init))
        pps (if input-pre-processors
              (pre-processor-helper input-pre-processors))
        bp-type (if backprop-type
                  (value-of-helper :backprop-type backprop-type))
        input-t (if input-type
                  (input-type-helper input-type))
        ;; updated config maps with code for creating java objs
        nn-conf-opts {:default-activation-fn a
                      :convolution-mode c-m
                      :default-dist d
                      :default-gradient-normalization g-norm
                      :default-learning-rate-policy lr-p
                      :optimization-algo o-a
                      :step-fn s-f
                      :default-updater u
                      :default-weight-init w}

        mln-conf-opts {:input-pre-processors pps
                       :backprop-type bp-type
                       :input-type input-t
                       :backprop? backprop?
                       :pretrain? pretrain?
                       :tbptt-back-length tbptt-back-length
                       :tbptt-fwd-length tbptt-fwd-length}

        ;; remove mln opts and layer opts
        ;; the mln methods should not be added until after the code
        ;; for the nn-conf builder is created
        ;; layers need to be treated after other nn-conf opts methods care created

        opts* (dissoc opts :layers :layer :backprop? :backprop-type
                      :input-pre-processors :input-type :pretrain? :tbptt-back-length
                      :tbptt-fwd-length :eval-and-build?)

        ;; map of methods to values/code to create objects
        updated-opts (replace-map-vals opts* nn-conf-opts)

        ;; use that map to set up the nn-conf builder code
        nn-conf-b (builder-fn nn-builder method-map-nn-builder updated-opts)

        ;; add in layers or just return the nn-conf builder
        builder-with-layers (if layers
                              (layer-builder-helper nn-conf-b layers)
                              nn-conf-b)
        ;; only pass the supplied method and values
        mln-conf-opts* (into {} (filter val mln-conf-opts))

        ;; create the code for creating the multi layer network conf
        mln-b (multi-layer-builder-helper mln-conf-opts* builder-with-layers layers)]
    ;; do we want to evaluate the code and build the result?
    (if eval-and-build?
      (eval-and-build mln-b)
      mln-b)))

(defn mln-from-nn-confs
  "creates a multi layer network from the supplied options and nn-confs

  the elelemts of the nn-confs data structure need to be homogeneous
   - all elelemts need to be code for creating and building nn-confs
     or the built dl4j java objects it can not be a mix of code and objects"
  [& {:keys [backprop? backprop-type input-pre-processors input-type pretrain?
             tbptt-back-length tbptt-fwd-length confs build?]
      :or {build? true}
      :as opts}]
  (let [b `(MultiLayerConfiguration$Builder.)

        pps (if input-pre-processors
              (pre-processor-helper input-pre-processors))
        bp-type (if backprop-type
                  (value-of-helper :backprop-type backprop-type))
        input-t (if input-type
                  (input-type-helper input-type))

        confz (match [(first confs)]
                     [(_ :guard seq?)] `(~into `() (map eval-and-build ~confs))
                     :else nil)

        mln-conf-opts {:input-pre-processors pps
                       :backprop-type bp-type
                       :input-type input-t
                       :backprop? backprop?
                       :pretrain? pretrain?
                       :tbptt-back-length tbptt-back-length
                       :tbptt-fwd-length tbptt-fwd-length
                       :conf confz}

        mln-conf-opts* (into {} (filter val mln-conf-opts))

        mln-b (builder-fn b multi-layer-methods mln-conf-opts*)

        mln-with-confs (if (not confz)
                         (.confs (eval mln-b) (into `() confs))
                         (eval mln-b))]
    (if build?
      (.build mln-with-confs)
      mln-with-confs)))
