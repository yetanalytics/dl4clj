(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/Layer.html"}
  dl4clj.nn.conf.layers.layer
  (:require [dl4clj.nn.conf.gradient-normalization :as gradient-normalization]
            [dl4clj.nn.conf.distribution.distribution :refer (distribution)]
            [dl4clj.nn.conf.updater :as updater]
            [dl4clj.nn.weights.weight-init :as weight-init])
  (:import [org.deeplearning4j.nn.conf.layers Layer$Builder]))

(defn builder [^Layer$Builder builder {:keys [activation ;; Layer activation function (String)
                                              activation-function ;; same as activation
                                              adam-mean-decay ;; Mean decay rate for Adam updater (double)
                                              adam-var-decay ;; Variance decay rate for Adam updater (double)
                                              bias-init      ;; (double) 
                                              dist ;; Distribution to sample initial weights from (Distribution or map)
                                              drop-out ;; (double) 
                                              gradient-normalization ;; Gradient normalization strategy (one of (dl4clj.nn.conf.gradient-normalization/values))
                                              gradient-normalization-threshold ;; Threshold for gradient normalization, only used for :clip-l2-per-layer, :clip-l2-per-param-type 
                                              ;; and clip-element-wise-absolute-value: L2 threshold for first two types of clipping, or absolute 
                                              ;; value threshold for last type of clipping
                                              l1       ;; L1 regularization coefficient (double)
                                              l2       ;; L2 regularization coefficient (double)
                                              learning-rate ;; (double)
                                              learning-rate-after ;; Learning rate schedule (java.util.Map<java.lang.Integer,java.lang.Double>)
                                              learning-rate-score-based-decay-rate ;; Rate to decrease learning-rate by when the score stops improving (double)
                                              momentum ;; Momentum rate (double)
                                              momentum-after ;; Momentum schedule. (java.util.Map<java.lang.Integer,java.lang.Double>)
                                              name ;; Layer name assigns layer string name (String)
                                              rho  ;; Ada delta coefficient (double)
                                              rms-decay ;; Decay rate for RMSProp (double)
                                              updater ;; Gradient updater  (one of (dl4clj.nn.conf.updater/values))
                                              weight-init ;; Weight initialization scheme (one of (dl4clj.nn.weights.weight-init/values))
                                              ]
                                       :or {}
                                       :as opts}]
  (when (or activation activation-function)
    (.activation builder (clojure.core/name (or activation activation-function))))
  (when adam-mean-decay
    (.adamMeanDecay builder adam-mean-decay))
  (when adam-var-decay
    (.adamVarDecay builder adam-var-decay))
  (when bias-init
    (.biasInit builder bias-init))
  (when dist
    (if (map? dist)
      (.dist builder (distribution dist))
      (.dist builder dist)))
  (when drop-out
    (.dropOut builder drop-out))
  (when gradient-normalization
    (.gradientNormalization builder (gradient-normalization/value-of gradient-normalization)))
  (when gradient-normalization-threshold
    (.gradientNormalizationThreshold builder gradient-normalization-threshold))
  (when l1
    (.l1 builder l1))
  (when l2
    (.l2 builder l2))
  (when learning-rate
    (.learningRate builder learning-rate))
  (when learning-rate-after
    (.learningRateAfter builder learning-rate-after))
  (when learning-rate-score-based-decay-rate
    (.learningRateScoreBasedDecayRate builder learning-rate-score-based-decay-rate))
  (when momentum
    (.momentum builder momentum))
  (when momentum-after
    (.momentumAfter builder momentum-after))
  (when name
    (.name builder name))
  (when rho
    (.rho builder rho))
  (when rms-decay
    (.rmsDecay builder rms-decay))
  (when updater
    (.updater builder (updater/value-of updater)))
  (when weight-init
    (.weightInit builder (weight-init/value-of weight-init))))


(defmulti layer (fn [opts] (first (keys opts))))
