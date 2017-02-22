(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/NeuralNetConfiguration.html"}
  dl4clj.nn.conf.neural-net-configuration
  (:require [dl4clj.nn.conf.layers.layer :as layer]
            [dl4clj.nn.conf.gradient-normalization :as gradient-normalization]
            [dl4clj.nn.conf.updater :as updater]
            [dl4clj.nn.api.optimization-algorithm :as opt]
            [dl4clj.nn.weights.weight-init :as weight-init]
            [dl4clj.nn.conf.distribution.distribution :as distribution]
            [clojure.data.json :as json]
            [dl4clj.nn.conf.multi-layer-configuration :as ml-config]
            [dl4clj.utils :refer (camel-to-dashed)])
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration NeuralNetConfiguration$Builder NeuralNetConfiguration$ListBuilder]
           ;; [org.nd4j.linalg.factory Nd4j]
           ))

(defn builder [{:keys [activation ;; Activation function / neuron non-linearity Typical values include :relu (rectified linear), :tanh, :sigmoid, :softmax, hardtanh, :leakyrelu, :maxout, :softsign, :softplus
                       activation-function ;; same as activation
                       adam-mean-decay ;; Mean decay rate for Adam updater (double) ;; broken
                       adam-var-decay  ;; Variance decay rate for Adam updater (double)
                       bias-init       ;; (double)
                       dist            ;; Distribution to sample initial weights from (Distribution)
                       drop-out        ;; (double)
                       gradient-normalization ;; Gradient normalization strategy (one of (dl4clj.nn.conf.gradient-normalization/values))
                       gradient-normalization-threshold ;; Threshold for gradient normalization, only used for :clip-l2-per-layer, :clip-l2-per-param-type
                       ;; and clip-element-wise-absolute-value: L2 threshold for first two types of clipping, or absolute
                       ;; value threshold for last type of clipping.

                       num-iterations      ;; Number of optimization iterations (int)
                       l1                  ;; L1 regularization coefficient (double)
                       l2                  ;; L2 regularization coefficient (double)
                       layer               ;; (Layer or layer configuration map)
                       learning-rate       ;; (double)
                       learning-rate-after ;; Learning rate schedule. Map of the iteration to the learning rate to apply at that iteration. ({integer,double})
                       learning-rate-score-based-decay-rate ;; Rate to decrease learningRate by when the score stops improving (double)
                       max-num-line-search-iterations       ;; (integer)
                       mini-batch ;; Process input as minibatch vs full dataset (boolean)
                       minimize ;; Objective function to minimize or maximize cost function. Default=true (boolean)
                       momentum ;; Momentum rate (double)
                       momentum-after    ;; Momentum schedule ({integer,double})
                       optimization-algo ;; (one of (dl4clj.nn.api.optimization-algorithm/values))
                       regularization ;; Whether to use regularization (l1, l2, dropout, etc.) (boolean)
                       use-regularization ;; same as regularization
                       rho            ;; Ada delta coefficient (double)
                       rms-decay      ;; Decay rate for RMSProp (double)
                       schedules ;; Whether to use schedules :learning-rate-after and :momentum-after (boolean)
                       seed      ;; Random number generator seed (int or long)
                       step-function ;; Step function to apply for back track line search (org.deeplearning4j.optimize.api.StepFunction)
                       updater       ;; Gradient updater (one of (dl4clj.nn.conf.updater/values))
                       use-drop-connect ;; Use drop connect: multiply the coefficients by a binomial sampling wrt the dropout probability (boolean)
                       weight-init ;; Weight initialization scheme (one of (dl4clj.nn.weights.weight-init/values))
                       ;; multi-layer parameters
                       list ;; Number of layers not including input (int)
                       backprop ;; Whether to do back prop or not (boolean)
                       backprop-type ;; (one of (backprop-type/values))
                       cnn-input-size ;; CNN input size, in order of [height,width,depth] (int-array)
                       confs ;; ([configuration maps])
                       input-pre-processors ;; ({integer,InputPreProcessor})
                       pretrain ;; Whether to do pre train or not (boolean)
                       redistribute-params ;; Whether to redistribute parameters as a view or not (boolean)
                       t-bptt-backward-length ;; When doing truncated BPTT: how many steps of backward should we do?
                       ;; Only applicable when doing backpropType(BackpropType.TruncatedBPTT)
                       ;; This is the k2 parameter on pg23 of http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf(int)
                       t-bptt-forward-length ;; When doing truncated BPTT: how many steps of forward pass should we do before doing (truncated) backprop? (int)
                       ;; Only applicable when doing backpropType(BackpropType.TruncatedBPTT)
                       ;; Typically tBPTTForwardLength parameter is same as the the tBPTTBackwardLength parameter, but may be larger than it in some circumstances (but never smaller)
                       ;; Ideally your training data time series length should be divisible by this This is the k1 parameter on pg23 of http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
                       ]
                :or {}
                :as opts}]
  (let [b (NeuralNetConfiguration$Builder.)]
    (when (or activation activation-function)
      (.activation b (name (or activation activation-function))))
    #_(when adam-mean-decay
      (.adamMeanDecay b adam-mean-decay))
    (when adam-var-decay
      (.adamVarDecay b adam-var-decay))
    (when bias-init
      (.biasInit b bias-init))
    (when dist
      (.dist b (if (map? dist) (distribution/distribution dist) dist)))
    (when drop-out
      (.dropOut b drop-out))
    (when gradient-normalization
      (.gradientNormalization b gradient-normalization))
    (when gradient-normalization-threshold
      (.gradientNormalizationThreshold b gradient-normalization-threshold))
    (when num-iterations
      (.iterations b num-iterations))
    (when l1
      (.l1 b l1))
    (when l2
      (.l2 b l2))
    (when layer
      (.layer b (if (map? layer) (layer/layer layer) layer)))
    (when learning-rate
      (.learningRate b learning-rate))
    (when learning-rate-after
      (.learningRateAfter b learning-rate-after))
    (when learning-rate-score-based-decay-rate
      (.learningRateScoreBasedDecayRate b learning-rate-score-based-decay-rate))
    (when max-num-line-search-iterations
      (.maxNumLineSearchIterations b max-num-line-search-iterations))
    (when (or mini-batch (contains? opts :mini-batch))
      (.miniBatch b mini-batch))
    (when (or minimize (contains? opts :minimize))
      (.minimize b minimize))
    (when momentum
      (.momentum b momentum))
    (when momentum-after
      (.momentumAfter b momentum-after))
    (when optimization-algo
      (.optimizationAlgo b (opt/value-of optimization-algo)))
    (when (or regularization use-regularization (contains? opts :regularization) (contains? opts :use-regularization))
      (.regularization b (or regularization use-regularization)))
    (when rho
      (.rho b rho))
    (when rms-decay
      (.rmsDecay b rms-decay))
    (when (or schedules (contains? opts :schedules))
      (.schedules b schedules))
    (when seed
      (.seed b seed))
    (when step-function
      (.stepFunction b step-function))
    (when updater
      (.updater b updater))
    (when (or use-drop-connect (contains? opts :use-drop-connect))
      (.useDropConnect b use-drop-connect))
    (when weight-init
      (.weightInit b weight-init))

    (cond list
          (ml-config/list-builder (.list b list)
                                  (:layers opts)
                                  opts)
          ;; confs ()
          :else b)))


(defn neural-net-configuration
  [opts]
  (.build (builder opts)))

(defn to-json [cfg]
  (.toJson cfg))
(defn from-json [^String  cfg]
  (NeuralNetConfiguration/fromJson cfg))

(defn to-edn [cfg]
  (json/read-str (.toJson cfg)
                 :key-fn #(keyword (camel-to-dashed %))))
(defn from-edn [cfg]
  (neural-net-configuration cfg))


(comment

  (require '[dl4clj.nn.conf.layers.graves-lstm :refer (graves-lstm)])
  (require '[dl4clj.nn.conf.layers.rnn-output-layer :refer (rnn-output)])
  (require '[dl4clj.nn.conf.distribution.uniform-distribution :refer (uniform-distribution)])
  (require '[dl4clj.nn.conf.distribution.binomial-distribution :refer (binomial-distribution)])
  (require '[dl4clj.nn.conf.distribution.normal-distribution :refer (normal-distribution)])
  (require '[dl4clj.nn.conf.neural-net-configuration :refer (neural-net-configuration)])
  (require '[nd4clj.linalg.lossfunctions.loss-functions :as loss-functions])
  (require '[dl4clj.nn.conf.distribution.distribution :refer (distribution)])
  (require '[dl4clj.nn.conf.multi-layer-configuration :as ml-configuration])

  (def opts {:optimization-algo :stochastic-gradient-descent
             :iterations 1
             :learning-rate 0.1
             :rms-decay 0.95
             :seed 12345
             :regularization true
             :l2 0.001
             :list 3
             :layers {0 {:graves-lstm {:n-in 50
                                       :n-out 100
                                       :updater :rmsprop
                                       :activation :tanh
                                       :weight-init :distribution
                                       :dist {:binomial {:number-of-trials 0, :probability-of-success 0.08}}}}
                      1 {:graves-lstm {:n-in 100
                                       :n-out 100
                                       :updater :rmsprop
                                       :activation :tanh
                                       :weight-init :distribution
                                       :dist {:uniform {:lower -0.08, :upper 0.08}}}}
                      2 {:rnnoutput {:loss-function :mcxent
                                     :n-in 100
                                     :n-out 50
                                     :activation :softmax
                                     :updater :rmsprop
                                     :weight-init :distribution
                                     :dist {:normal {:mean 0.0, :std 0.05}}}}}
             :pretrain false
             :backprop true})

  (def cfg (neural-net-configuration opts))

  (to-edn cfg)

  (def opts2 (update-in (to-edn cfg) [:confs] #(map neural-net-configuration %)))


  (def cfg2-builder
    (ml-configuration/builder opts2))

  (def cfg2 (.build cfg2-builder))

  (= (to-edn cfg) (to-edn cfg2))





  (def opts {:optimization-algo :stochastic-gradient-descent
             :iterations 1
             :learning-rate 0.1
             :use-regularization true
             :drop-out 0.5

             :momentum 0.99
             :rms-decay 0.95

             :seed 12345

             :list 3
             :layers {0 {:graves-lstm {:n-in 50
                                       :n-out 100
                                       :updater :rmsprop ;; :adagrad
                                       :activation :tanh
                                       :weight-init :distribution
                                       :dist {:normal {:mean 0.0, :std 0.1}}
                                       :forget-gate-bias-init 1.0}}
                      1 {:graves-lstm {:n-in 100
                                       :n-out 100
                                       :updater :rmsprop
                                       :activation :tanh
                                       :weight-init :distribution
                                       :dist {:normal {:mean 0.0, :std 0.1}}
                                       :forget-gate-bias-init 1.0}}
                      2 {:rnnoutput {:loss-function :mcxent
                                     :n-in 100
                                     :n-out 50
                                     :activation :softmax
                                     :updater :rmsprop
                                     :weight-init :distribution
                                     :dist {:normal {:mean 0.0, :std 0.1}}}}}
             :pretrain false
             :backprop true})




  (map neural-net-configuration (:confs opts2))



  (def cfg (neural-net-configuration opts))


  (neural-net-configuration (to-edn cfg))
  (print (to-json cfg) )

  (print (.toJson cfg))
  ;;; ???
  ;;; - edn doesn't show the layers
  ;;; - is layer needed (check in example)



  )
