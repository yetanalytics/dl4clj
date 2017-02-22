(ns dl4clj.nn.conf.builders.nn-conf-builder
  (:require [dl4clj.nn.conf.distribution.distribution :as distribution]
            [dl4clj.nn.conf.builders.builders :as layer-builders]
            [dl4clj.nn.conf.builders.multi-layer-builders :as multi-layer])
  (:import [org.deeplearning4j.nn.conf
            NeuralNetConfiguration$Builder
            NeuralNetConfiguration$ListBuilder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           ))
(defn nn-conf-builder
  [{:keys [activation-fn
           adam-mean-decay
           adam-var-decay
           bias-init
           bias-learning-rate
           dist
           drop-out
           epsilon
           gradient-normalization
           gradient-normalization-threshold
           l1
           l2
           layer
           layers
           leaky-relu-alpha
           learning-rate
           learning-rate-policy
           learning-rate-schedule
           lr-policy-decay-rate
           lr-policy-power
           lr-policy-steps
           lr-score-based-decay
           max-num-line-search-iterations
           mini-batch
           minimize
           momentum
           momentum-schedule
           num-iterations
           optimization-algo
           rho
           rms-decay
           seed
           step-fn
           updater
           use-drop-connect
           use-regularization
           weight-init]
    :or {}
    :as opts}]
  (let [b (NeuralNetConfiguration$Builder.)]
    (if (contains? opts :activation-fn)
      (.activation b activation-fn) b)
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
    #_(if (contains? opts :epsilon)
      (.epsilon b epsilon) b)
    (if (contains? opts :gradient-normalization)
      (.gradientNormalization b gradient-normalization) b)
    (if (contains? opts :gradient-normalization-threshold)
      (.gradientNormalizationThreshold b gradient-normalization-threshold) b)
    (if (contains? opts :l1)
      (.l1 b l1) b)
    (if (contains? opts :l2)
      (.l2 b l2) b)
    #_(if (contains? opts :layer)
      (.layer b (layer-builders/builder b layer)) b)













    (if (contains? opts :layers) ;; this needs to be last so all the other config is already done
      (multi-layer/list-builder b (:layers opts)) b)
    b
    #_(.build b)))
(.build (nn-conf-builder {
                  :layers {0 {:dense-layer {:n-in 100
                                            :n-out 1000
                                            :layer-name "first layer"}}
                           1 {:output-layer {:layer-name "second layer"
                                             :n-in 1000
                                             :n-out 10
                                             }}}}))
