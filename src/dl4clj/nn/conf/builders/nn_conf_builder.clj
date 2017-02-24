(ns dl4clj.nn.conf.builders.nn-conf-builder
  (:require [dl4clj.nn.conf.distribution.distribution :as distribution]
            [dl4clj.nn.conf.builders.builders :as layer-builders]
            [dl4clj.nn.conf.builders.multi-layer-builders :as multi-layer])
  (:import [org.deeplearning4j.nn.conf
            NeuralNetConfiguration$Builder
            NeuralNetConfiguration$ListBuilder
            ComputationGraphConfiguration$GraphBuilder
            LearningRatePolicy
            GradientNormalization
            Updater]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.nd4j.linalg.activations Activation]
           ))
(defn nn-conf-builder
  [{:keys [global-activation-fn
           adam-mean-decay
           adam-var-decay
           bias-init
           bias-learning-rate
           dist
           drop-out
           epsilon
           gradient-normalization
           gradient-normalization-threshold
           graph-builder
           iterations
           l1
           l2
           layer
           layers
           leaky-relu-alpha
           learning-rate
           learning-rate-policy
           learning-rate-schedule
           learning-rate-score-based-decay-rate
           lr-policy-decay-rate
           lr-policy-power
           lr-policy-steps
           max-num-line-search-iterations
           mini-batch
           minimize
           momentum
           momentum-after
           optimization-algo
           regularization
           rho
           rms-decay
           seed
           step-fn
           updater
           use-drop-connect
           weight-init
           ]
    :or {}
    :as opts}]
  (let [b (NeuralNetConfiguration$Builder.)]
    (if (contains? opts :global-activation-fn)
      (.activation b (Activation/valueOf global-activation-fn)) b) ;; make a value of string coversion fn
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
    (if (contains? opts :drop-out)
      (.dropOut b drop-out) b)
    (if (contains? opts :epsilon)
      (.epsilon b epsilon) b)
    (if (contains? opts :gradient-normalization)
      (.gradientNormalization
       b
       (GradientNormalization/valueOf gradient-normalization))
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
       (LearningRatePolicy/valueOf learning-rate-policy)) b)
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
      (.optimizationAlgo b (OptimizationAlgorithm/valueOf optimization-algo)) b)
    (if (contains? opts :regularization)
      (.regularization b regularization) b)
    (if (contains? opts :rho)
      (.rho b rho) b)
    (if (contains? opts :rms-decay)
      (.rmsDecay b rms-decay) b)
    (if (contains? opts :seed)
      (.seed b seed) b)
    #_(if (contains? opts :step-fn)
        (.stepFunction b )) ;; figure out step fns
    (if (contains? opts :updater)
      (.updater b (Updater/valueOf updater)) b)
    (if (contains? opts :use-drop-connect)
      (.useDropConnect b use-drop-connect) b)
    (if (contains? opts :weight-init)
      (.weightInit b (WeightInit/valueOf weight-init)) b)
    (if (contains? opts :layer)
      (.layer b (layer-builders/builder layer)) b)
    (if (contains? opts :layers) ;; this needs to be last so all the other config is already done
      (multi-layer/list-builder b layers) b)
    ;; look into how to use the graph builder
    ))


#_(nn-conf-builder {:global-activation-fn "RELU"
                     :layers {0 {:dense-layer {:n-in 100
                                               :n-out 1000
                                               :layer-name "first layer"
                                               :activation-fn "TANH"}}
                              1 {:output-layer {:layer-name "second layer"
                                                :n-in 1000
                                                :n-out 10
                                                }}}})
