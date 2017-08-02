(ns dl4clj.nn.builders.nn-builder
  (:require [dl4clj.nn.conf.builders.builders :as layer-builders]
            [dl4clj.helpers :refer [value-of-helper
                                    distribution-helper
                                    step-fn-helper
                                    pre-processor-helper
                                    input-type-helper]]
            [dl4clj.utils :refer [builder-fn replace-map-vals eval-and-build]]
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

        opts* (dissoc opts :layers :layer :backprop? :backprop-type :input-pre-processors
                      :input-type :pretrain? :tbptt-back-length :tbptt-fwd-length)

        mln-conf-opts* (into {} (filter val mln-conf-opts))

        ;; map of methods to values/code to create objects
        updated-opts (replace-map-vals opts* nn-conf-opts)

        ;; use that map to set up the nn-conf builder code
        nn-conf-b (builder-fn nn-builder method-map-nn-builder updated-opts)

        ;; add in layers or just return the nn-conf builder
        builder-with-layers (if layers
                              ;; did the user pass us any layers to create?
                              (if (keyword? (first (keys layers)))
                                ;; only a single layer
                                `(.layer ~nn-builder (eval-and-build (layer-builders/builder ~layers)))
                                ;; multiple layers
                                (builder-fn `(.list ~nn-conf-b) {:add-layers '.layer}
                                            {:add-layers
                                             (into [] (for [each layers
                                                            :let [[idx layer] each]]
                                                        [idx `(eval-and-build (layer-builders/builder ~layer))]))}))
                              ;; no layers
                              nn-conf-b)]
    (eval-and-build
     ;; refactor condition tree
     (cond (keyword? (first (keys layers)))
          ;; if we only had one layer, need to use a mln builder to add mln opts
          (builder-fn `(MultiLayerConfiguration$Builder.) multi-layer-methods
                      (assoc mln-conf-opts* :conf `(~list (eval-and-build ~builder-with-layers))))
          ;; if we had multiple layers, evaled code will create the multi-layer-conf builder
          (integer? (first (keys layers)))
          (builder-fn builder-with-layers multi-layer-methods mln-conf-opts*)
          ;; if we didnt get passed any mln-conf methods, just return the builder with layers added
          (empty? mln-conf-opts*) builder-with-layers
          :else
          ;; we were just passed options for setting up a 0 layer nn-conf and options for setting up a mln
          ;; it is assumed that the user will add layers to this mln later
          (builder-fn `(MultiLayerConfiguration$Builder.) multi-layer-methods
                      (assoc mln-conf-opts* :conf `(~list (eval-and-build ~builder-with-layers))))))))

#_(nn-builder :default-activation-fn :relu
            :step-fn :negative-gradient-step-fn
            :default-updater :none
            :use-drop-connect? true
            :default-drop-out 0.2
            :default-weight-init :xavier-uniform
            :build? false
            :default-gradient-normalization :renormalize-l2-per-layer
            #_:layers #_{:activation-layer {:n-in 1000
                                            :n-out 10
                                            :layer-name "second layer"
                                            :activation-fn :tanh
                                            :gradient-normalization :none}}

            #_{0 {:activation-layer {:n-in 100
                                     :n-out 1000
                                     :layer-name "first layer"
                                     :activation-fn :tanh
                                     :gradient-normalization :none}}
               1 {:activation-layer {:n-in 1000
                                     :n-out 10
                                     :layer-name "second layer"
                                     :activation-fn :tanh
                                     :gradient-normalization :none}}}
            ;;:backprop-type :standard
            ;;:pretrain? true
            ;;:backprop? true
            )
;; TODO
;; make a fn for mln from confs
