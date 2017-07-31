(ns dl4clj.nn.builders.nn-builder
  (:require [dl4clj.nn.conf.builders.builders :as layer-builders]
            [dl4clj.helpers :refer [value-of-helper
                                    distribution-helper
                                    step-fn-helper
                                    #_layer-builder-helper
                                    pre-processor-helper
                                    input-type-helper]]
            [dl4clj.utils :refer [builder-fn replace-map-vals eval-and-build]])
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
   :default-weight-init                      '.weightInit
   :layer                                    '.layer
   :layers                                   '.layer
   ;; multi layer methods
   :backprop?                                '.backprop
   :backprop-type                            '.backpropType
   :input-pre-processors                     '.inputPreProcessors
   :input-type                               '.setInputType
   :pretrain?                                '.pretrain
   :tbptt-back-length                        '.tBPTTBackwardLength
   :tbptt-fwd-length                         '.tBPTTForwardLength})

(defn nn-builder
  ""
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
             tbptt-back-length tbptt-fwd-length]
      :or {nn-builder `(NeuralNetConfiguration$Builder.)}
      :as opts}]
  (let [a (if default-activation-fn
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
        nn-conf-opts {:default-activation-fn a
                      :convolution-mode c-m
                      :default-dist d
                      :default-gradient-normalization g-norm
                      :default-learning-rate-policy lr-p
                      :optimization-algo o-a
                      :step-fn s-f
                      :default-updater u
                      :default-weight-init w}
        ;; builder with args
        updated-opts (replace-map-vals (dissoc opts :layers :layer) nn-conf-opts)
        nn-conf-b (builder-fn nn-builder method-map-nn-builder updated-opts)


        ;; the layer or layers
        l (if layer
            (builder-fn nn-conf-b {:add-layer '.layer}
                        {:add-layer `(eval-and-build (layer-builders/builder ~layer))}))

        ls (if layers
             (builder-fn `(.list ~nn-conf-b) {:add-layers '.layer}
                         {:add-layers (into [] (for [each layers
                                                     :let [[idx layer] each]]
                                                 [idx `(eval-and-build (layer-builders/builder ~layer))]))}))

        mln-conf-opts {:input-pre-processors pps
                       :backprop-type bp-type
                       :input-type input-t}

        ]

    (if l
      l
      ls)
    #_(test-layers nn-conf-b layers)

    #_(if layer
      (.layer (eval (builder-fn nn-builder method-map-nn-builder updated-opts))
              (.build (eval (layer-builders/builder layer)))))

    #_(eval (layer-builders/builder layer))

    #_(.layer (eval (builder-fn nn-builder method-map-nn-builder updated-opts))
            l)))


;; go down and refactor the layer level
(eval-and-build (nn-builder :default-activation-fn :relu
            :step-fn :negative-gradient-step-fn
            :default-updater :none
            :use-drop-connect? true
            :default-drop-out 0.2
            :default-weight-init :xavier-uniform
            :build? false
            :default-gradient-normalization :renormalize-l2-per-layer
            #_:layers #_{0 {:activation-layer {:n-in 100
                                           :n-out 1000
                                           :layer-name "first layer"
                                           :activation-fn :tanh
                                           :gradient-normalization :none}}
                     1 {:activation-layer {:n-in 1000
                                           :n-out 10
                                           :layer-name "second layer"
                                           :activation-fn :tanh
                                           :gradient-normalization :none}}}
            :layer {:activation-layer {:n-in 100
                                           :n-out 1000
                                           :layer-name "first layer"
                                           :activation-fn :tanh
                                           :gradient-normalization :none}}))

(defn test-layers
  [nn-conf-builder layers]
  (let [b `(.list ~nn-conf-builder)
        calls (into [] (for [each layers
                             :let [[idx layer] each]]
                         [idx `(layer-builders/builder ~layer)]))]
    (builder-fn b {:add-layers '.layer}
                {:add-layers calls})))

(.build (eval (test-layers `(NeuralNetConfiguration$Builder.) {0 {:dense-layer {:n-in 100
                                                                                :n-out 1000
                                                                                :layer-name "first layer"
                                                                                :activation-fn :tanh
                                                                                :gradient-normalization :none}}
                                                               1 {:dense-layer {:n-in 1000
                                                                                :n-out 10
                                                                                :layer-name "second layer"
                                                                                :activation-fn :tanh
                                                                                :gradient-normalization :none}}})))

#_(multi-layer-config-builder
 :list-b (nn-conf-builder :default-activation-fn :relu
                                 :step-fn :negative-gradient-step-fn
                                 :default-updater :none
                                 :use-drop-connect? true
                                 :default-drop-out 0.2
                                 :default-weight-init :xavier-uniform
                                 :build? false
                                 :default-gradient-normalization :renormalize-l2-per-layer
                                 :layers {0 {:dense-layer {:n-in 100
                                                           :n-out 1000
                                                           :layer-name "first layer"
                                                           :activation-fn :tanh
                                                           :gradient-normalization :none}}
                                          1 {:dense-layer {:n-in 1000
                                                           :n-out 10
                                                           :layer-name "second layer"
                                                           :activation-fn :tanh
                                                           :gradient-normalization :none}}})
 ;;:nn-confs nn-conf
 :backprop? true
 :build? false
 :backprop-type :standard
 :input-pre-processors {0 {:zero-mean-pre-pre-processor {}}
                        1 {:unit-variance-processor {}}}
 :input-type {:feed-forward {:size 100}}
 :pretrain? false)
