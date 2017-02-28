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
;; implement and add documentation for graph builder
(defn nn-conf-builder
  "given a map of parameters and their values, generates the appropriate
  builder for constructing a network.  The params provided here function
  as the default params for a layer.  Specifying a param within the layer config maps
  will be the param used, not the one supplied here.

  Params are:

  :global-activation-fn (keyword) default global activation fn,
  wont overwrite activation functions specificed in the layer or layers map.
   opts are :cube, :elu, :hard-sigmoid, :hard-tanh, :identity, :leaky-relu
            :relu, :r-relu, :sigmoid, :soft-max, :soft-plus, :soft-sign, :tanh

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

  :iterations (int) Number of optimization iterations

  :l1 (double) L1 regularization coefficient

  :l2 (double) L2 regularization coefficient used when regularization is set to true

  :layer {:layer-type {config-opts vals}} map of layer config params.
    see dl4clj.nn.conf.builders.builders
    :layer will be ignored if :layers is supplied, layer only creates a single layer

  :layers {idx (int) {:layer-type {config-opts vals}}}
   see dl4clj.nn.conf.builders.builders

  :leaky-relu-alpha (double) f(x) = alpha*x for x < 0
   Special version of a Rectified Linear Unit that allows a small gradient when the unit is not active

  :learning-rate (double) Paramter that controls the learning rate

  :learning-rate-policy (keyword) How to decay learning rate during training
   see https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/LearningRatePolicy.html
   one of :none, :exponential, :inverse, :poly, :sigmoid, :step, :torch-step :schedule :score

  :learning-rate-schedule {int double} Map of the iteration to the learning rate to apply at that iteration

  :learning-rate-score-based-decay-rate (double)
   Rate to decrease :learning-rate by when the score stops improving

  :lr-policy-decay-rate (double) Sets the decay rate for the learning rate decay policy.

  :lr-policy-power (double) Sets the power used for learning rate inverse policy

  :lr-policy-steps (double) Sets the number of steps used for learning decay rate steps policy

  :max-num-line-search-iterations (int) Maximum number of line search iterations

  :mini-batch (boolean) Process input as minibatch instead of as a full dataset

  :minimize (boolean) Objective function to minimize or maximize cost function
   default is true

  :momentum (double) Momentum rate used only when the :updater is set to :nesterovs

  :momentum-after {int double} Map of the iteration to the momentum rate to apply at that iteration
   also only used when :updater is :nesterovs

  :optimization-algo (keyword) Optimization algorithm to use
   one of: :line-gradient-descent, :conjugate-gradient, :hessian-free,
           :lbfgs, :stochastic-gradient-descent

  :regularization (boolean) Whether or not to use regularization

  :rho (double) Ada delta coefficient

  :rms-decay (double) Decay rate for RMSProp, only applies if using :updater :RMSPROP

  :seed (int or long) Random number generator seed, Used for reproducability between runs

  :step-fn (keyword) Step functions to apply for back track line search
   Only applies for line search optimizers: Line Search SGD, Conjugate Gradient,
   LBFGS Options: :default-step-fn (default),
   :negative-default-step-fn :gradient-step-fn (for SGD),
   :negative-gradient-step-fn

  :updater (keyword) Gradient updater,
   one of: :adagrad, :sgd, :adam, :adadelta, :nesterovs, :adagrad, :rmsprop, :none, :custom

  :use-drop-connect (boolean) Multiply the weight by a binomial sampling wrt the dropout probability.
   Dropconnect probability is set using :drop-out(double); this is the probability of retaining a weight

  :weight-init (keyword) Weight initialization scheme
  see https://deeplearning4j.org/doc/org/deeplearning4j/nn/weights/WeightInit.html (use cases)
  one of: :distribution, :zero, :sigmoid-uniform, :uniform, :xavier, :xavier-uniform
          :xavier-fan-in, :xavier-legacy, :relu, :relu-uniform, :vi, :size, :normalized
"
  [{:keys [global-activation-fn adam-mean-decay adam-var-decay bias-init
           bias-learning-rate dist drop-out epsilon gradient-normalization
           gradient-normalization-threshold iterations l1 l2 layer layers
           leaky-relu-alpha learning-rate learning-rate-policy
           learning-rate-schedule learning-rate-score-based-decay-rate
           lr-policy-decay-rate lr-policy-power lr-policy-steps
           max-num-line-search-iterations mini-batch minimize
           momentum momentum-after optimization-algo regularization
           rho rms-decay seed step-fn updater use-drop-connect
           weight-init
           ;;graph-builder
           ]
    :or {}
    :as opts}]
  (let [b (NeuralNetConfiguration$Builder.)]
    (if (contains? opts :global-activation-fn)
      (.activation b (constants/value-of {:activation-fn global-activation-fn})) b)
    (if (contains? opts :adam-mean-decay)
      (.adamMeanDecay b adam-mean-decay) b)
    (if (contains? opts :adam-var-decay)
      (.adamVarDecay b adam-var-decay) b)
    (if (contains? opts :bias-init)
      (.biasInit b bias-init) b)
    (if (contains? opts :bias-learning-rate)
      (.biasLearningRate b bias-learning-rate) b)
    (if (contains? opts :dist)
      (.dist b (if (map? dist)
               (distribution/distribution dist)
               dist)) b)
    (if (contains? opts :epsilon)
      (.epsilon b epsilon) b)
    (if (contains? opts :gradient-normalization)
      (.gradientNormalization
       b
       (constants/value-of {:gradient-normalization gradient-normalization}))
      b)
    (if (contains? opts :gradient-normalization-threshold)
      (.gradientNormalizationThreshold b gradient-normalization-threshold) b)
    (if (contains? opts :iterations)
      (.iterations b iterations) b)
    (if (contains? opts :l1)
      (.l1 b l1) b)
    (if (contains? opts :l2)
      (.l2 b l2) b)
    (if (contains? opts :leaky-relu-alpha)
      (.leakyreluAlpha b leaky-relu-alpha) b)
    (if (contains? opts :learning-rate)
      (.learningRate b learning-rate) b)
    (if (contains? opts :learning-rate-policy)
      (.learningRateDecayPolicy
       b
       (constants/value-of {:learning-rate-policy learning-rate-policy})) b)
    (if (contains? opts :learning-rate-schedule)
      (.learningRateSchedule b learning-rate-schedule) b)
    (if (contains? opts :learning-rate-score-based-decay-rate)
      (.learningRateScoreBasedDecayRate b learning-rate-score-based-decay-rate) b)
    (if (contains? opts :lr-policy-decay-rate)
      (.lrPolicyDecayRate b lr-policy-decay-rate) b)
    (if (contains? opts :lr-policy-power)
      (.lrPolicyPower b lr-policy-power) b)
    (if (contains? opts :lr-policy-steps)
      (.lrPolicySteps b lr-policy-steps) b)
    (if (contains? opts :max-num-line-search-iterations)
      (.maxNumLineSearchIterations b max-num-line-search-iterations) b)
    (if (contains? opts :mini-batch)
      (.miniBatch b mini-batch) b)
    (if (contains? opts :minimize)
      (.minimize b minimize) b)
    (if (contains? opts :momentum)
      (.momentum b momentum) b)
    (if (contains? opts :momentum-after)
      (.momentumAfter b momentum-after) b)
    (if (contains? opts :optimization-algo)
      (.optimizationAlgo
       b
       (constants/value-of {:optimization-algorithm optimization-algo}))
      b)
    (if (contains? opts :regularization)
      (.regularization b regularization) b)
    (if (contains? opts :rho)
      (.rho b rho) b)
    (if (contains? opts :rms-decay)
      (.rmsDecay b rms-decay) b)
    (if (contains? opts :seed)
      (.seed b seed) b)
    (if (contains? opts :step-fn)
      (.stepFunction b (step-functions/step-fn step-fn)) b)
    (if (contains? opts :updater)
      (.updater b (constants/value-of {:updater updater})) b)
    (if (contains? opts :use-drop-connect)
      (if (and (true? use-drop-connect) (contains? opts :drop-out))
        (-> (.useDropConnect b use-drop-connect)
            (.dropOut drop-out))
        (.useDropConnect b use-drop-connect))
      b)
    (if (contains? opts :weight-init)
      (.weightInit b (constants/value-of {:weight-init weight-init})) b)
    (if (contains? opts :layer)
      (.layer b (layer-builders/builder layer)) b)
    (if (contains? opts :layers) ;; this needs to be last so all the other config is already done
      (multi-layer/list-builder b layers) b)
    ;; look into how to use the graph builder
    ))


(comment
  (println
   (str
    (.build
     (nn-conf-builder {:global-activation-fn "RELU"
                       :step-fn :negative-gradient-step-fn
                       :updater :none
                       :use-drop-connect true
                       :drop-out 0.2
                       :weight-init :xavier-uniform
                       :gradient-normalization :renormalize-l2-per-layer
                       :layers {0 {:dense-layer {:n-in 100
                                                 :n-out 1000
                                                 :layer-name "first layer"
                                                 :activation-fn "TANH"
                                                 :gradient-normalization :none }}
                                1 {:output-layer {:layer-name "second layer"
                                                  :n-in 1000
                                                  :n-out 10
                                                  }}}})))))
