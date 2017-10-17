(ns dl4clj.nn-tests
  (:require [dl4clj.nn.conf.builders.layers :as layer]
            [dl4clj.nn.conf.builders.nn :as nn]
            [dl4clj.nn.conf.distributions :refer :all]
            [dl4clj.nn.conf.input-pre-processor :refer :all]
            [dl4clj.nn.api.input-type :refer :all]
            [dl4clj.nn.conf.step-fns :refer :all]
            [dl4clj.nn.api.variational-distribution :refer :all]
            [dl4clj.nn.api.classifier :as classifier]
            [dl4clj.nn.api.distribution :as dist]
            [dl4clj.nn.api.input-pre-processors :as ipp]
            [dl4clj.nn.api.layer :as layer-api]
            [dl4clj.nn.api.layer-specific-fns :as lsf]
            [dl4clj.nn.api.model :as model]
            [dl4clj.nn.api.multi-layer-conf :as multi-layer-conf]
            [dl4clj.nn.api.multi-layer-network :as mln]
            [dl4clj.nn.api.nn-conf :as nn-conf]
            [dl4clj.constants :refer :all]
            [dl4clj.nn.multilayer.multi-layer-network :as multi-layer-network]
            [dl4clj.nn.transfer-learning.fine-tune-conf :as fine-tune-conf]
            [dl4clj.nn.transfer-learning.helper :as tl-helper]
            [dl4clj.nn.api.fine-tune :refer :all]
            [dl4clj.nn.transfer-learning.transfer-learning :as tl]
            [dl4clj.nn.gradient.default-gradient :as gradient]
            [dl4clj.nn.layer-creation :as layer-creation]
            [dl4clj.nn.api.gradient :as grad]

            ;; helper fns
            [dl4clj.utils :refer [array-of get-labels eval-and-build as-code]]
            [nd4clj.linalg.factory.nd4j :refer [indarray-of-zeros]]
            [dl4clj.datasets.default-datasets :refer [new-mnist-ds]]
            [dl4clj.datasets.iterators :refer [new-mnist-data-set-iterator]]
            [dl4clj.datasets.api.datasets :refer [get-features get-example]]
            [dl4clj.eval.evaluation :refer [new-classification-evaler]]
            [dl4clj.eval.api.eval :refer [get-stats]]
            [cheshire.core :refer [parse-string]]
            [clojure.test :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; helper fn for layer creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn quick-nn-conf
  [layer]
  (nn/builder :optimization-algo     :stochastic-gradient-descent
              :iterations            1
              :default-learning-rate 0.006
              :lr-policy-decay-rate  0.2
              :build?                true
              :layers                layer))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; distributions to sample weights from
;; dl4clj.nn.conf.distribution.distribution
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest distributions-test
  (testing "the creation of distributions"
    ;; uniform
    (is (= '(org.deeplearning4j.nn.conf.distribution.UniformDistribution. 0.2 0.4)
           (new-uniform-distribution :lower 0.2 :upper 0.4)))
    (is (= org.deeplearning4j.nn.conf.distribution.UniformDistribution
           (type (new-uniform-distribution :lower 0.2 :upper 0.4 :as-code? false))))
    ;; normal
    (is (= org.deeplearning4j.nn.conf.distribution.NormalDistribution
           (type (new-normal-distribution :mean 0 :std 1 :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.distribution.NormalDistribution. 0 1)
           (new-normal-distribution :mean 0 :std 1)))

    ;; guassian (same as normal)
    (is (= org.deeplearning4j.nn.conf.distribution.GaussianDistribution
           (type (new-gaussian-distribution :mean 0.0 :std 1 :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.distribution.GaussianDistribution. 0.0 1)
           (new-gaussian-distribution :mean 0.0 :std 1)))
    ;; binomial
    (is (= org.deeplearning4j.nn.conf.distribution.BinomialDistribution
           (type (new-binomial-distribution :probability-of-success 0.5
                                            :number-of-trials 1
                                            :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.distribution.BinomialDistribution. 1 0.5)
           (new-binomial-distribution :probability-of-success 0.5
                                      :number-of-trials 1)))

    ;; bernoulli
    (is (= org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution
           (type (new-bernoulli-reconstruction-distribution :activation-fn :sigmoid :as-code? false))))
    (is (= org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution
           (type (new-bernoulli-reconstruction-distribution :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution.
             (dl4clj.constants/value-of {:activation-fn :sigmoid}))
           (new-bernoulli-reconstruction-distribution :activation-fn :sigmoid)))

    ;; exponential reconstruction
    (is (= org.deeplearning4j.nn.conf.layers.variational.ExponentialReconstructionDistribution
           (type (new-exponential-reconstruction-distribution :activation-fn :relu :as-code? false))))
    (is (= org.deeplearning4j.nn.conf.layers.variational.ExponentialReconstructionDistribution
           (type (new-exponential-reconstruction-distribution :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.layers.variational.ExponentialReconstructionDistribution.
             (dl4clj.constants/value-of {:activation-fn :relu}))
           (new-exponential-reconstruction-distribution :activation-fn :relu)))

    ;; gaussian reconstruction
    (is (= org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution
           (type (new-gaussian-reconstruction-distribution :activation-fn :relu :as-code? false))))
    (is (= org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution
           (type (new-gaussian-reconstruction-distribution :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution.
             (dl4clj.constants/value-of {:activation-fn :relu}))
           (new-gaussian-reconstruction-distribution :activation-fn :relu)))

    ;; composite
    (is (= org.deeplearning4j.nn.conf.layers.variational.CompositeReconstructionDistribution
           (type (eval (distribution
                        {:composite
                         {:distributions
                          [{:bernoulli {:activation-fn :sigmoid
                                        :dist-size     5}}
                           {:exponential {:activation-fn :sigmoid
                                          :dist-size     3}}
                           {:gaussian-reconstruction {:activation-fn :hard-tanh
                                                      :dist-size     1}}
                           {:bernoulli {:activation-fn :sigmoid
                                        :dist-size     4}}]}})))))
    (is (= '(dl4clj.nn.conf.distributions/distribution
             {:composite
              {:distributions
               [{:bernoulli {:activation-fn :sigmoid, :dist-size 5}}
                {:exponential {:activation-fn :sigmoid, :dist-size 3}}
                {:gaussian-reconstruction {:activation-fn :hard-tanh, :dist-size 1}}
                {:bernoulli {:activation-fn :sigmoid, :dist-size 4}}]}})
           (new-composite-reconstruction-distribution
            :distributions
            [{:bernoulli {:activation-fn :sigmoid
                          :dist-size     5}}
             {:exponential {:activation-fn :sigmoid
                            :dist-size     3}}
             {:gaussian-reconstruction {:activation-fn :hard-tanh
                                        :dist-size     1}}
             {:bernoulli {:activation-fn :sigmoid
                          :dist-size     4}}])))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; step functions for use in nn-conf creation
;; dl4clj.nn.conf.step-fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest step-fn-test
  (testing "the creation of step fns"
    (is (= '(org.deeplearning4j.nn.conf.stepfunctions.DefaultStepFunction.)
           (new-default-step-fn)))
    (is (= org.deeplearning4j.nn.conf.stepfunctions.DefaultStepFunction
           (type (new-default-step-fn :as-code? false))))

    (is (= '(org.deeplearning4j.nn.conf.stepfunctions.GradientStepFunction.)
           (new-gradient-step-fn)))
    (is (= org.deeplearning4j.nn.conf.stepfunctions.GradientStepFunction
           (type (new-gradient-step-fn :as-code? false))))

    (is (= org.deeplearning4j.nn.conf.stepfunctions.NegativeDefaultStepFunction
           (type (new-negative-default-step-fn :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.stepfunctions.NegativeDefaultStepFunction.)
           (new-negative-default-step-fn)))

    (is (= org.deeplearning4j.nn.conf.stepfunctions.NegativeGradientStepFunction
           (type (new-negative-gradient-step-fn :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.stepfunctions.NegativeGradientStepFunction.)
           (new-negative-gradient-step-fn)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; input pre-processors test
;; dl4clj.nn.conf.input-pre-processor
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest pre-processors-test
  (testing "the creation of input preprocessors for use in multi-layer-conf"
    ;; binominal sampling pre-processor
    (is (= org.deeplearning4j.nn.conf.preprocessor.BinomialSamplingPreProcessor
           (type (new-binominal-sampling-pre-processor :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.preprocessor.BinomialSamplingPreProcessor.)
           (new-binominal-sampling-pre-processor)))

    ;; unit variance processor
    (is (= org.deeplearning4j.nn.conf.preprocessor.UnitVarianceProcessor
           (type (new-unit-variance-processor :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.preprocessor.UnitVarianceProcessor.)
           (new-unit-variance-processor)))

    ;; Rnn -> Cnn pre-processor
    (is (= org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor
           (type (new-rnn-to-cnn-pre-processor :input-height 1 :input-width 1
                                               :num-channels 1 :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor. 1 1 1)
           (new-rnn-to-cnn-pre-processor :input-height 1 :input-width 1
                                         :num-channels 1)))

    ;; zero mean and unit variance
    (is (= org.deeplearning4j.nn.conf.preprocessor.ZeroMeanAndUnitVariancePreProcessor
           (type (new-zero-mean-and-unit-variance-pre-processor :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.preprocessor.ZeroMeanAndUnitVariancePreProcessor.)
           (new-zero-mean-and-unit-variance-pre-processor)))

    ;; zero mean pre-pre processor
    (is (= org.deeplearning4j.nn.conf.preprocessor.ZeroMeanPrePreProcessor
           (type (new-zero-mean-pre-pre-processor :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.preprocessor.ZeroMeanPrePreProcessor.)
           (new-zero-mean-pre-pre-processor)))

    ;; cnn -> feed foward pre processor
    (is (= org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor
           (type (new-cnn-to-feed-forward-pre-processor :input-height 1
                                                        :input-width 1
                                                        :num-channels 1
                                                        :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor. 1 1 1)
           (new-cnn-to-feed-forward-pre-processor :input-height 1
                                                  :input-width 1
                                                  :num-channels 1)))

    ;; cnn -> rnn pre processor
    (is (= org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor
           (type (new-cnn-to-rnn-pre-processor :input-height 1 :input-width 1
                                               :num-channels 1 :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor. 1 1 1)
           (new-cnn-to-rnn-pre-processor :input-height 1 :input-width 1
                                         :num-channels 1)))

    ;; feed forward -> cnn pre processor
    (is (= org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor
           (type (new-feed-forward-to-cnn-pre-processor :input-height 1
                                                        :input-width 1
                                                        :num-channels 1
                                                        :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor. 1 1 1)
           (new-feed-forward-to-cnn-pre-processor :input-height 1
                                                  :input-width 1
                                                  :num-channels 1)))

    ;; rnn -> feed forward pre processor
    (is (= org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor
           (type (new-rnn-to-feed-forward-pre-processor :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor.)
           (new-rnn-to-feed-forward-pre-processor)))

    ;; feed forward -> rnn pre processor
    (is (= org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor
           (type (new-feed-forward-to-rnn-pre-processor :as-code? false))))
    (is (= '(org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor.)
           (new-feed-forward-to-rnn-pre-processor)))

    ;; combine multiple pre-processors
    (is (= '(org.deeplearning4j.nn.conf.preprocessor.ComposableInputPreProcessor.
             (dl4clj.utils/array-of
              :data [(org.deeplearning4j.nn.conf.preprocessor.ZeroMeanPrePreProcessor.)
                     (org.deeplearning4j.nn.conf.preprocessor.BinomialSamplingPreProcessor.)]
              :java-type org.deeplearning4j.nn.conf.InputPreProcessor))
           (new-composable-input-pre-processor
            :components [(new-zero-mean-pre-pre-processor)
                             (new-binominal-sampling-pre-processor)])))
    (is (= org.deeplearning4j.nn.conf.preprocessor.ComposableInputPreProcessor
           (type (new-composable-input-pre-processor
                  :components [(new-zero-mean-pre-pre-processor)
                               (new-binominal-sampling-pre-processor)]
                  :as-code? false))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; creation of default gradients
;; dl4clj.nn.gradient.default-gradient
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest default-gradient-test
  (testing "the creation and manipulation of gradients"
    (let [grad-with-var (grad/set-gradient-for!
                         :as-code? false
                         :grad (gradient/new-default-gradient)
                         :variable "foo"
                         :new-gradient (indarray-of-zeros
                                        :as-code? true
                                        :rows 2 :columns 2))]
      (is (= org.deeplearning4j.nn.gradient.DefaultGradient
             (type (gradient/new-default-gradient :as-code? false))))
      (is (= org.deeplearning4j.nn.gradient.DefaultGradient
             (type grad-with-var)))
      ;; I don't think this test is reliable bc it assumes cpu
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (grad/gradient :grad grad-with-var))))
      (is (= java.util.LinkedHashMap
             (type (grad/gradient-for-variable grad-with-var))))
      ;; gradient order was not explictly set
      (is (= nil
             (type (grad/flattening-order-for-variables :grad grad-with-var
                                                        :variable "foo")))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; nn-conf-builder
;; dl4clj.nn.conf.builders.nn
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest setting-vals
  (let [parsed-conf (parse-string
                     (multi-layer-conf/to-json
                      (nn/builder
                       ;; vals for the first layer
                       :default-activation-fn :relu
                       :default-updater :nesterovs
                       :default-weight-init :xavier-uniform
                       :default-bias-init 0.2
                       :default-bias-learning-rate 0.2
                       :default-drop-out 0.2
                       :default-gradient-normalization :renormalize-l2-per-layer
                       :default-gradient-normalization-threshold 0.3
                       :default-l1 0.5
                       :default-l1-bias 0.2
                       :default-l2 0.8
                       :default-l2-bias 0.2
                       :default-learning-rate 0.3
                       :default-learning-rate-policy :poly
                       ;; ^ this is the learning rate policy for all layers
                       ;; nn vals
                       :iterations 2
                       :lr-policy-decay-rate 0.4
                       :lr-policy-power 0.4
                       :max-num-line-search-iterations 10
                       :mini-batch? true
                       :minimize? true
                       :use-drop-connect? true
                       :optimization-algo :line-gradient-descent
                       :lr-score-based-decay-rate 0.001
                       :regularization? true
                       :seed 123
                       :step-fn :negative-gradient-step-fn
                       :build? true
                       :as-code? false
                       :layers {0 (layer/dense-layer-builder :n-in 100
                                                             :n-out 1000
                                                             :layer-name "first layer"
                                                             :momentum 0.4
                                                             :momentum-after {0 0.2 1 0.4})
                                1 {:dense-layer {:n-in 1000
                                                 :n-out 2
                                                 :activation-fn :tanh
                                                 :adam-mean-decay 0.2
                                                 :adam-var-decay 0.1
                                                 :bias-init 0.7
                                                 :bias-learning-rate 0.1
                                                 :dist {:normal {:mean 0 :std 1}}
                                                 :drop-out 0.2
                                                 :epsilon 0.3
                                                 :gradient-normalization :none
                                                 :gradient-normalization-threshold 0.9
                                                 :l1 0.2
                                                 :l1-bias 0.1
                                                 :l2 0.4
                                                 :l2-bias 0.3
                                                 :layer-name "second layer"
                                                 :learning-rate 0.1
                                                 :learning-rate-schedule {0 0.2 1 0.5}
                                                 :updater :adam
                                                 :weight-init :distribution}}}
                       ;;mln vals
                       :backprop? true
                       :backprop-type :standard
                       :input-pre-processors {0 (new-zero-mean-pre-pre-processor)}
                       :pretrain? false)) true)
        nn-conf (:confs parsed-conf)
        ;; a nn conf is created for each layer behind the scene (in java world)
        [conf-l1 conf-l2] nn-conf
        l1 (:dense (:layer conf-l1))
        l2 (:dense (:layer conf-l2))]
    ;; see args for nn/builder
    ;; mln args
    (testing "the multi layer network args set"
      (is (= true (:backprop parsed-conf)))
      (is (= "Standard" (:backpropType parsed-conf)))
      (is (= {:0 {:zeroMean {}}} (:inputPreProcessors parsed-conf)))
      (is (= false (:pretrain parsed-conf))))
    ;; nn args
    (testing "the nn-conf args set"
      (is (= "Poly" (:learningRatePolicy conf-l1) (:learningRatePolicy conf-l2)))
      (is (= 0.4 (:lrPolicyDecayRate conf-l1) (:lrPolicyDecayRate conf-l2)))
      (is (= true (:miniBatch conf-l1) (:miniBatch conf-l2)))
      (is (= 10 (:maxNumLineSearchIterations conf-l1) (:maxNumLineSearchIterations conf-l2)))
      (is (= 2 (:numIterations conf-l1) (:numIterations conf-l2)))
      (is (= 0.4 (:lrPolicyPower conf-l1) (:lrPolicyPower conf-l2)))
      (is (= true (:useDropConnect conf-l1) (:useDropConnect conf-l2)))
      (is (= true (:useRegularization conf-l1) (:useRegularization conf-l2)))
      (is (= 123 (:seed conf-l1) (:seed conf-l2))))
    ;; layer args (defaults set for l1)
    (testing "the default layer 1 args set"
      (is (= 100 (:nin l1)))
      (is (= 1000 (:nout l1)))
      (is (= {:ReLU {}} (:activationFn l1)))
      (is (= "RenormalizeL2PerLayer" (:gradientNormalization l1)))
      (is (= "first layer" (:layerName l1)))
      (is (= 0.2 (:l2Bias l1)))
      (is (= 0.4 (:momentum l1)))
      (is (= "NESTEROVS" (:updater l1)))
      (is (= 0.8 (:l2 l1)))
      (is (= {:0 0.2, :1 0.4} (:momentumSchedule l1)))
      (is (= "XAVIER_UNIFORM" (:weightInit l1)))
      (is (= 0.2 (:biasInit l1)))
      (is (= 0.3 (:learningRate l1)))
      (is (= 0.2 (:l1Bias l1)))
      (is (= 0.5 (:l1 l1)))
      (is (= 0.3 (:gradientNormalizationThreshold l1)))
      (is (= 0.3 (:biasLearningRate l1))))
    (testing "the explictly set layer 2 args"
      (is (= 1000 (:nin l2)))
      (is (= 2 (:nout l2)))
      (is (= {:TanH {}} (:activationFn l2)))
      (is (= 0.2 (:adamMeanDecay l2)))
      (is (= 0.1 (:adamVarDecay l2)))
      (is (= 0.7 (:biasInit l2)))
      (is (= 0.1 (:biasLearningRate l2)))
      (is (= {:normal {:mean 0.0, :std 1.0}} (:dist l2)))
      (is (= 0.2 (:dropOut l2)))
      (is (= 0.3 (:epsilon l2)))
      (is (= "None" (:gradientNormalization l2)))
      (is (= 0.9 (:gradientNormalizationThreshold l2)))
      (is (= 0.2 (:l1 l2)))
      (is (= 0.1 (:l1Bias l2)))
      (is (= 0.4 (:l2 l2)))
      (is (= 0.3 (:l2Bias l2)))
      (is (= "second layer" (:layerName l2)))
      (is (= 0.1 (:learningRate l2)))
      (is (= {:0 0.2, :1 0.5} (:learningRateSchedule l2)))
      (is (= "ADAM" (:updater l2)))
      (is (= "DISTRIBUTION" (:weightInit l2))))))

(deftest nn-test
  (testing "the helper fns for builder"
    ;; layer builder helper
    (is (= '(.layer "im a builder" (dl4clj.utils/eval-and-build (str "im a layer created by a fn")))
           (nn/layer-builder-helper "im a builder" '(str "im a layer created by a fn"))))
    (is (= '(.layer
             "im a builder"
             (dl4clj.utils/eval-and-build
              (dl4clj.nn.conf.builders.layers/builder
               {:some-single-layer {:some-layer-config "some config value"}})))
           (nn/layer-builder-helper
            "im a builder"
            {:some-single-layer {:some-layer-config "some config value"}})))
    (is (= '(doto (.list "im a builder")
              (.layer
               0
               (dl4clj.utils/eval-and-build
                (dl4clj.nn.conf.builders.layers/builder
                 {:first-layer {:some-config "some value"}})))
              (.layer
               1
               (dl4clj.utils/eval-and-build
                (dl4clj.nn.conf.builders.layers/builder
                 {:second-layer {:other-config "other value"}}))))
           (nn/layer-builder-helper
            "im a builder"
            {0 {:first-layer {:some-config "some value"}}
             1 {:second-layer {:other-config "other value"}}})))
    (is (= '(doto (.list "im a builder")
              (.layer
               0
               (dl4clj.utils/eval-and-build
                (im-a-fn-call "with" "some" "args")))
              (.layer
               1
               (dl4clj.utils/eval-and-build
                (dl4clj.nn.conf.builders.layers/builder
                 {:layer-2 {:some-config "some value"}}))))
           (nn/layer-builder-helper
            "im a builder"
            {0 '(im-a-fn-call "with" "some" "args")
             1 {:layer-2 {:some-config "some value"}}})))
    (is (= '(doto (.list "im a builder")
              (.layer
               0
               (dl4clj.utils/eval-and-build
                (im-a-fn-call "with" "some" "args")))
              (.layer
               1
               (dl4clj.utils/eval-and-build
                (im-another-fn-call :keyword "args"))))
           (nn/layer-builder-helper
            "im a builder"
            {0 '(im-a-fn-call "with" "some" "args")
             1 '(im-another-fn-call :keyword "args")})))
    ;; multi layer builder helper
    ;; no mln opts supplied, just returns the builder passed
    (is (= "some builder"
           (nn/multi-layer-builder-helper {} "some builder" "layers")))
    ;; we use the layers to determine what builder to use
    (is (= '(doto "some list builder" (.backprop true))
           (nn/multi-layer-builder-helper {:backprop? true}
                                          "some list builder"
                                          {0 {:dense-layer {:n-in 10}}
                                           1 (layer/dense-layer-builder :n-in 10)})))
    (is (= '(.build
             (doto (org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder.)
               (.stepFunction (org.deeplearning4j.nn.conf.stepfunctions.GradientStepFunction.))
               (.regularization true)
               (.learningRateDecayPolicy (dl4clj.constants/value-of {:learning-rate-policy :poly}))
               (.seed 123) (.maxNumLineSearchIterations 6) (.lrPolicyDecayRate 0.3)
               (.useDropConnect false) (.minimize true) (.iterations 1)
               (.miniBatch true) (.learningRateScoreBasedDecayRate 0.7)
               (.optimizationAlgo (dl4clj.constants/value-of {:optimization-algorithm :lbfgs}))
               (.convolutionMode (dl4clj.constants/value-of {:convolution-mode :strict}))
               (.lrPolicyPower 0.1)))
           (nn/builder
            :iterations 1
            :lr-policy-decay-rate 0.3
            :max-num-line-search-iterations 6
            :mini-batch? true
            :minimize? true
            :use-drop-connect? false
            :optimization-algo :lbfgs
            :lr-score-based-decay-rate 0.7
            :regularization? true
            :seed 123
            :step-fn :gradient-step-fn
            :convolution-mode :strict
            :lr-policy-power 0.1
            :default-learning-rate-policy :poly
            :build? true
            :as-code? true)))
    (is (= '(.build
             (doto
                 (doto
                     (.list
                      (doto
                          (org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder.)
                        (.l2 0.8)
                        (.gradientNormalizationThreshold 0.3)
                        (.stepFunction
                         (org.deeplearning4j.nn.conf.stepfunctions.NegativeGradientStepFunction.))
                        (.weightInit
                         (dl4clj.constants/value-of {:weight-init :xavier-uniform}))
                        (.regularization true)
                        (.gradientNormalization
                         (dl4clj.constants/value-of
                          {:gradient-normalization :renormalize-l2-per-layer}))
                        (.learningRateDecayPolicy
                         (dl4clj.constants/value-of {:learning-rate-policy :poly}))
                        (.updater (dl4clj.constants/value-of {:updater :nesterovs}))
                        (.l1 0.5)
                        (.biasLearningRate 0.2)
                        (.dropOut 0.2)
                        (.seed 123)
                        (.maxNumLineSearchIterations 10)
                        (.lrPolicyDecayRate 0.4)
                        (.l1Bias 0.2)
                        (.useDropConnect true)
                        (.minimize true)
                        (.iterations 2)
                        (.learningRate 0.3)
                        (.biasInit 0.2)
                        (.miniBatch true)
                        (.activation (dl4clj.constants/value-of {:activation-fn
                                                                 :relu}))
                        (.learningRateScoreBasedDecayRate 0.001)
                        (.optimizationAlgo
                         (dl4clj.constants/value-of
                          {:optimization-algorithm :line-gradient-descent}))
                        (.lrPolicyPower 0.4)
                        (.l2Bias 0.2)))
                   (.layer
                    0
                    (dl4clj.utils/eval-and-build
                     (doto
                         (org.deeplearning4j.nn.conf.layers.DenseLayer$Builder.)
                       (.nOut 1000)
                       (.momentumAfter {0 0.2, 1 0.4})
                       (.nIn 100)
                       (.momentum 0.4)
                       (.name "first layer"))))
                   (.layer
                    1
                    (dl4clj.utils/eval-and-build
                     (dl4clj.nn.conf.builders.layers/builder
                      {:dense-layer
                       {:learning-rate-policy :inverse,
                        :l1-bias 0.1,
                        :l1 0.2,
                        :drop-out 0.2,
                        :n-out 2,
                        :activation-fn :tanh,
                        :dist {:normal {:mean 0, :std 1}},
                        :gradient-normalization :none,
                        :bias-learning-rate 0.1,
                        :weight-init :distribution,
                        :adam-var-decay 0.1,
                        :bias-init 0.7,
                        :n-in 1000,
                        :l2-bias 0.3,
                        :l2 0.4,
                        :updater :adam,
                        :learning-rate-schedule {0 0.2, 1 0.5},
                        :epsilon 0.3,
                        :layer-name "foo11",
                        :learning-rate 0.1,
                        :adam-mean-decay 0.2,
                        :gradient-normalization-threshold 0.9}}))))
               (.inputPreProcessors
                {0
                 (org.deeplearning4j.nn.conf.preprocessor.ZeroMeanPrePreProcessor.)})
               (.backpropType
                (dl4clj.constants/value-of {:backprop-type :standard}))
               (.backprop true)))
         (nn/builder
            ;; vals for the first layer
            :default-activation-fn :relu
            :default-updater :nesterovs
            :default-weight-init :xavier-uniform
            :default-bias-init 0.2
            :default-bias-learning-rate 0.2
            :default-drop-out 0.2
            :default-gradient-normalization :renormalize-l2-per-layer
            :default-gradient-normalization-threshold 0.3
            :default-l1 0.5
            :default-l1-bias 0.2
            :default-l2 0.8
            :default-l2-bias 0.2
            :default-learning-rate 0.3
            :default-learning-rate-policy :poly
            ;; nn vals
            :iterations 2
            :lr-policy-decay-rate 0.4
            :lr-policy-power 0.4
            :max-num-line-search-iterations 10
            :mini-batch? true
            :minimize? true
            :use-drop-connect? true
            :optimization-algo :line-gradient-descent
            :lr-score-based-decay-rate 0.001
            :regularization? true
            :seed 123
            :step-fn :negative-gradient-step-fn
            :build? true
            :as-code? true
            :layers {0 (layer/dense-layer-builder :n-in 100
                                                  :n-out 1000
                                                  :layer-name "first layer"
                                                  :momentum 0.4
                                                  :momentum-after {0 0.2 1 0.4})
                     1 {:dense-layer {:n-in 1000
                                      :n-out 2
                                      :activation-fn :tanh
                                      :adam-mean-decay 0.2
                                      :adam-var-decay 0.1
                                      :bias-init 0.7
                                      :bias-learning-rate 0.1
                                      :dist {:normal {:mean 0 :std 1}}
                                      :drop-out 0.2
                                      :epsilon 0.3
                                      :gradient-normalization :none
                                      :gradient-normalization-threshold 0.9
                                      :l1 0.2
                                      :l1-bias 0.1
                                      :l2 0.4
                                      :l2-bias 0.3
                                      :layer-name "foo11"
                                      :learning-rate 0.1
                                      :learning-rate-policy :inverse
                                      :learning-rate-schedule {0 0.2 1 0.5}
                                      :updater :adam
                                      :weight-init :distribution}}}
            ;;mln vals
            :backprop? true
            :backprop-type :standard
            :input-pre-processors {0 (new-zero-mean-pre-pre-processor)}
            :pretrain? false)))
    ;; here we have to use the constructor for the MultiLayerConfigBuilder
    ;; because we don't have a list builder, we just have a nn-conf
    (let [layer-fn-call-code (nn/multi-layer-builder-helper
                              {:backprop? true}
                              "a nn conf builder"
                              (layer/dense-layer-builder :n-in 10))
          layer-config-code (nn/multi-layer-builder-helper
                             {:backprop? true}
                             "a nn conf builder"
                             {:dense-layer {:n-in 10}})
          nil-layers (nn/multi-layer-builder-helper
                      {:backprop? true}
                      "a nn conf builder"
                      nil)
          [_ mln-builder] layer-fn-call-code
          [_ mln-b] layer-config-code
          [_ mln] nil-layers
          ]
      (is (= '(org.deeplearning4j.nn.conf.MultiLayerConfiguration$Builder.)
             mln-builder))
      (is (= '(org.deeplearning4j.nn.conf.MultiLayerConfiguration$Builder.)
             mln-b))
      (is (= '(org.deeplearning4j.nn.conf.MultiLayerConfiguration$Builder.)
             mln))))
  (testing "the creation of neural network configurations"
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
           (type
            (nn/builder
             :iterations 1
             :lr-policy-decay-rate 0.3
             :max-num-line-search-iterations 6
             :mini-batch? true
             :minimize? true
             :use-drop-connect? false
             :optimization-algo :lbfgs
             :lr-score-based-decay-rate 0.7
             :regularization? true
             :seed 123
             :step-fn :gradient-step-fn
             :convolution-mode :strict
             :lr-policy-power 0.1
             :default-learning-rate-policy :poly
             :build? true
             :as-code? false))))
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
           (type
            (nn/builder
             :iterations 1
             :lr-policy-decay-rate 0.3
             :lr-policy-power 0.4
             :default-learning-rate-policy :poly
             :max-num-line-search-iterations 6
             :mini-batch? true
             :minimize? true
             :use-drop-connect? false
             :optimization-algo :lbfgs
             :lr-score-based-decay-rate 0.7
             :regularization? true
             :seed 123
             :step-fn :default-step-fn
             :convolution-mode :strict
             :build? false
             :as-code? false))))
    (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
           (type
            (nn/builder :default-activation-fn :relu
                        :step-fn :negative-gradient-step-fn
                        :default-updater :none
                        :use-drop-connect? true
                        :default-drop-out 0.2
                        :default-weight-init :xavier-uniform
                        :build? true
                        :default-gradient-normalization :renormalize-l2-per-layer
                        :as-code? false
                        :layers {0 {:dense-layer {:n-in 100
                                                  :n-out 1000
                                                  :layer-name "first layer"
                                                  :activation-fn :tanh
                                                  :gradient-normalization :none}}
                                 1 {:dense-layer {:n-in 1000
                                                  :n-out 10
                                                  :layer-name "second layer"
                                                  :gradient-normalization :none}}}))))
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration$ListBuilder
           (type
            (nn/builder :default-activation-fn :relu
                        :step-fn :negative-gradient-step-fn
                        :default-updater :none
                        :use-drop-connect? true
                        :default-drop-out 0.2
                        :default-weight-init :xavier-uniform
                        :build? false
                        :default-gradient-normalization :renormalize-l2-per-layer
                        :as-code? false
                        :layers {0 {:dense-layer {:n-in 100
                                                  :n-out 1000
                                                  :layer-name "first layer"
                                                  :activation-fn :tanh
                                                  :gradient-normalization :none}}
                                 1 {:dense-layer {:n-in 1000
                                                  :n-out 10
                                                  :layer-name "second layer"
                                                  :activation-fn :tanh
                                                  :gradient-normalization :none}}}))))
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
           ;; show that the activation fn changes
           (type
            (nn/builder :default-activation-fn :relu
                        :step-fn :negative-gradient-step-fn
                        :default-updater :none
                        :use-drop-connect? true
                        :default-drop-out 0.2
                        :default-weight-init :xavier-uniform
                        :default-gradient-normalization :renormalize-l2-per-layer
                        :build? true
                        :as-code? false
                        :layers {:dense-layer {:n-in 100
                                               :n-out 1000
                                               :layer-name "first layer"
                                               :activation-fn :tanh
                                               :gradient-normalization :none}}))))
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
           ;; show that fns can be passed
           (type
            (nn/builder :default-activation-fn :relu
                        :step-fn :negative-gradient-step-fn
                        :default-updater :none
                        :use-drop-connect? true
                        :default-drop-out 0.2
                        :default-weight-init :xavier-uniform
                        :default-gradient-normalization :renormalize-l2-per-layer
                        :build? true
                        :as-code? false
                        :layers (layer/dense-layer-builder :n-in 100
                                                           :n-out 1000
                                                           :layer-name "first layer"
                                                           :activation-fn :tanh
                                                           :gradient-normalization :none)))))
    ;; we can also pass our multi layer args to nn/builder for single or multi layer confs
    (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
         (type
          (nn/builder :default-activation-fn :relu
                      :step-fn :negative-gradient-step-fn
                      :default-updater :none
                      :use-drop-connect? true
                      :default-drop-out 0.2
                      :default-weight-init :xavier-uniform
                      :default-gradient-normalization :renormalize-l2-per-layer
                      :build? true
                      :as-code? false
                      :layers {:dense-layer {:n-in 100
                                             :n-out 1000
                                             :layer-name "first layer"
                                             :activation-fn :tanh
                                             :gradient-normalization :none}}
                      ;; multi layer args
                      :backprop? true
                      :input-pre-processors {0 {:zero-mean-pre-pre-processor {}}
                                             1 (new-unit-variance-processor)}))))
    (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
           (type
            (nn/builder :default-activation-fn :relu
                        :step-fn :negative-gradient-step-fn
                        :default-updater :none
                        :use-drop-connect? true
                        :default-drop-out 0.2
                        :default-weight-init :xavier-uniform
                        :default-gradient-normalization :renormalize-l2-per-layer
                        :build? true
                        :as-code? false
                        ;; multi layer arg
                        :backprop? true
                        :layers {0 {:dense-layer {:n-in 100
                                                  :n-out 1000
                                                  :layer-name "first layer"
                                                  :activation-fn :tanh
                                                  :gradient-normalization :none}}
                                 1 (layer/dense-layer-builder :n-in 10)}))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi-layer-config-builder test
;; dl4clj.nn.conf.builders.multi-layer-builders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest multi-layer-builder-test
  (testing "the creation of mutli layer nn's from single layer confs"
    (let [nn-conf (nn/builder :default-activation-fn :relu
                              :step-fn :negative-gradient-step-fn
                              :default-updater :none
                              :use-drop-connect? true
                              :default-drop-out 0.2
                              :default-weight-init :xavier-uniform
                              :default-gradient-normalization :renormalize-l2-per-layer
                              :build? false
                              :as-code? true
                              :layers {:dense-layer {:n-in 10
                                                     :n-out 100
                                                     :layer-name "another layer"
                                                     :activation-fn :tanh
                                                     :gradient-normalization :none}})]
      ;; with a list builder, a built nn-conf, and all opts
      (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
             (type
              (nn/mln-from-nn-confs :confs [nn-conf]
                                    :backprop? true
                                    :backprop-type :standard
                                    :input-pre-processors {0 {:zero-mean-pre-pre-processor {}}
                                                           1 {:unit-variance-processor {}}}
                                    :input-type {:feed-forward {:size 100}}
                                    :pretrain? false
                                    :build? true))))
      (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
             (type
              (nn/mln-from-nn-confs :confs [nn-conf nn-conf]
                                    :backprop? true
                                    :backprop-type :standard
                                    :input-pre-processors {0 {:zero-mean-pre-pre-processor {}}
                                                           1 {:unit-variance-processor {}}}
                                    :input-type {:feed-forward {:size 100}}
                                    :pretrain? false
                                    :build? true))))
      (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
             (type
              (nn/mln-from-nn-confs :confs [nn-conf nn-conf]
                                    :build? true)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi-layer-network test
;; dl4clj.nn.multilayer.multi-layer-network
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest multi-layer-creation-test
  (testing "the creation of multi layer networks from objects and from code"
    (let [mln-conf-code (nn/builder :seed 123
                                    :optimization-algo :stochastic-gradient-descent
                                    :iterations 1
                                    :default-learning-rate 0.006
                                    :default-updater :nesterovs
                                    :default-momentum 0.9
                                    :regularization? true
                                    :default-l2 1e-4
                                    :build? true
                                    :default-gradient-normalization :renormalize-l2-per-layer
                                    :layers {0 (layer/dense-layer-builder
                                                :n-in 784
                                                :n-out 1000
                                                :layer-name "first layer"
                                                :activation-fn :relu
                                                :weight-init :xavier)
                                             1 {:output-layer {:n-in 1000
                                                               :n-out 10
                                                               :layer-name "second layer"
                                                               :activation-fn :soft-max
                                                               :weight-init :xavier}}}
                                    :backprop? true
                                    :pretrain? false)

          mln-conf-obj (eval mln-conf-code)

          mln-as-code (multi-layer-network/new-multi-layer-network :conf mln-conf-code)

          mln-from-obj (multi-layer-network/new-multi-layer-network :conf mln-conf-obj)]
      (is (= '(org.deeplearning4j.nn.multilayer.MultiLayerNetwork.
               (.build
                (doto
                    (doto
                        (.list
                         (doto
                             (org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder.)
                           (.l2 1.0E-4)
                           (.regularization true)
                           (.gradientNormalization
                            (dl4clj.constants/value-of
                             {:gradient-normalization :renormalize-l2-per-layer}))
                           (.updater (dl4clj.constants/value-of {:updater :nesterovs}))
                           (.seed 123)
                           (.momentum 0.9)
                           (.iterations 1)
                           (.learningRate 0.006)
                           (.optimizationAlgo
                            (dl4clj.constants/value-of
                             {:optimization-algorithm :stochastic-gradient-descent}))))
                      (.layer
                       0
                       ;; fn call to create layer
                       (dl4clj.utils/eval-and-build
                        (doto
                            (org.deeplearning4j.nn.conf.layers.DenseLayer$Builder.)
                          (.nOut 1000)
                          (.activation (dl4clj.constants/value-of {:activation-fn :relu}))
                          (.weightInit (dl4clj.constants/value-of {:weight-init :xavier}))
                          (.nIn 784)
                          (.name "first layer"))))
                      (.layer
                       1
                       ;; pass config map to create layer
                       (dl4clj.utils/eval-and-build
                        (dl4clj.nn.conf.builders.layers/builder
                         {:output-layer
                          {:n-in 1000,
                           :n-out 10,
                           :layer-name "second layer",
                           :activation-fn :soft-max,
                           :weight-init :xavier}}))))
                  (.backprop true))))
             mln-as-code))
      (is (= (type mln-from-obj) (type (eval mln-as-code)))))))

(deftest multi-layer-network-method-test
  (testing "multi layer network methods"
    (let [mln-conf (nn/builder :seed 123
                               :optimization-algo :stochastic-gradient-descent
                               :iterations 1
                               :default-learning-rate 0.006
                               :default-updater :nesterovs
                               :default-momentum 0.9
                               :regularization? true
                               :default-l2 1e-4
                               :build? true
                               :default-gradient-normalization :renormalize-l2-per-layer
                               :layers {0 (layer/dense-layer-builder
                                           :n-in 784
                                           :n-out 1000
                                           :layer-name "first layer"
                                           :activation-fn :relu
                                           :weight-init :xavier)
                                        1 {:output-layer {:n-in 1000
                                                          :n-out 10
                                                          :layer-name "second layer"
                                                          :activation-fn :soft-max
                                                          :weight-init :xavier}}}
                               :backprop? true
                               :pretrain? false
                               :as-code? false)
          mln (multi-layer-network/new-multi-layer-network :conf mln-conf)
          init-mln (mln/initialize! :mln mln :ds (new-mnist-ds :as-code? false))
          mnist-iter (new-mnist-data-set-iterator :batch-size 128 :train? true :seed 123 :as-code? false)
          input (get-features (get-example :ds (new-mnist-ds :as-code? false) :idx 0))
          eval (new-classification-evaler :n-classes 10 :as-code? false)
          init-no-ds (model/init! :model mln)
          evaled (mln/evaluate-classification :mln init-no-ds :iter mnist-iter)
          _ (println "\n example evaluation stats \n")
          _ (println (get-stats :evaler evaled))]
      ;;other-input (get-mln-input :mln init-mln)
      ;;^ this currently crashes all of emacs
      ;; need a fitted mln for (is (= "" (type (get-epsilon :mln ...))))
      ;; might need a fitted mln for this too (fine-tune! :mln init-no-ds)
      ;; was getting an unexpected error: Mis matched lengths: [600000] != [10]
      ;; need to init a mln with a recurrent layer to test
      ;; rnn-... fns
      (is (= org.deeplearning4j.nn.multilayer.MultiLayerNetwork (type mln)))
      ;; these tests will have to be updated to include other NDArray types so wont break with gpus
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (mln/activate-selected-layers
                    :from 0 :to 1 :mln mln :input input))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (mln/activate-from-prev-layer
                    :current-layer-idx 0 :mln mln :input input :training? true))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (mln/activate-from-prev-layer
                    :current-layer-idx 1 :mln mln :training? true
                    ;; input to second layer is output of first
                    :input (mln/activate-from-prev-layer
                            :current-layer-idx 0 :mln mln :input input :training? true)))))
      (is (= org.deeplearning4j.nn.multilayer.MultiLayerNetwork
             (type (mln/clear-layer-mask-arrays! mln))))
      (is (= java.util.ArrayList
             (type (mln/compute-z :mln init-mln :training? true :input input))))
      (is (= java.util.ArrayList
             (type (mln/compute-z :mln init-mln :training? true))))
      (is (= java.lang.String (type (get-stats :evaler evaled))))
      (is (= org.deeplearning4j.eval.Evaluation
             (type (mln/evaluate-classification :mln init-no-ds :iter mnist-iter))))
      (is (= org.deeplearning4j.eval.Evaluation
             (type
              (mln/evaluate-classification :mln init-no-ds :iter mnist-iter
                                       :labels-list ["0" "1" "2" "3" "4" "5" "6" "7" "8" "9"]))))
      (is (= org.deeplearning4j.eval.RegressionEvaluation
             (type (mln/evaluate-regression :mln init-no-ds :iter mnist-iter))))
      (is (= java.util.ArrayList
             (type (mln/feed-forward :mln init-mln :input input))))
      (is (= java.util.ArrayList
             (type (mln/feed-forward :mln init-mln))))
      (is (= java.util.ArrayList (type (mln/feed-forward-to-layer :mln init-mln :layer-idx 0 :train? true))))
      (is (= java.util.ArrayList (type (mln/feed-forward-to-layer :mln init-mln :layer-idx 0 :input input))))
      (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
             (type (mln/get-default-config init-mln))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray (type (mln/get-input init-mln))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray (type (get-labels init-mln))))
      (is (= org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer
             (type (mln/get-layer :mln init-mln :layer-idx 0))))
      (is (= ["first layer" "second layer"] (mln/get-layer-names mln)))
      (is (= (type (array-of :data [] :java-type org.deeplearning4j.nn.api.Layer))
             (type (mln/get-layers init-mln))))
      (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
             (type (mln/get-layer-wise-config init-mln))))
      ;; we never set a mask
      (is (= nil (mln/get-mask init-mln)))
      (is (= 2 (mln/get-n-layers init-mln)))
      (is (= org.deeplearning4j.nn.layers.OutputLayer (type (mln/get-output-layer init-mln))))
      (is (= org.deeplearning4j.nn.updater.MultiLayerUpdater (type (mln/get-updater init-mln))))
      (is (= (type mln) (type (mln/initialize-layers! :mln mln :input input))))
      (is (true? (mln/is-init-called? mln)))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (mln/output :mln init-mln :input input))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (mln/output :mln init-no-ds :iter mnist-iter))))
      (is (= (type mln) (type (mln/print-config mln))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (mln/reconstruct :mln mln
                                    :layer-output (first (mln/feed-forward-to-layer
                                                          :layer-idx 0 :mln mln :input input))
                                    :layer-idx 1))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (mln/score-examples :mln init-no-ds :iter mnist-iter
                                   :add-regularization-terms? false))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (mln/score-examples :mln init-mln :dataset (new-mnist-ds :as-code? false)
                                   :add-regularization-terms? false))))
      (is (= java.lang.String (type (mln/summary init-mln))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (mln/z-from-prev-layer :mln init-mln :input input
                                          :current-layer-idx 0 :training? true)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fine tuning/transfer learning
;; dl4clj.nn.transfer-learning.fine-tune-conf
;; dl4clj.nn.transfer-learning.helper
;; dl4clj.nn.transfer-learning.transfer-learning
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest transfer-learning-tests
  (testing "the transfer learning fns"
    (let [;; nn conf code and obj
          nn-conf-code (nn/builder :default-activation-fn :relu
                                   :step-fn :negative-gradient-step-fn
                                   :default-updater :none
                                   :use-drop-connect? false
                                   :default-weight-init :xavier-uniform
                                   :default-gradient-normalization :none
                                   :layers {:dense-layer {:n-in 10
                                                          :n-out 100
                                                          :layer-name "some layer"
                                                          :activation-fn :tanh
                                                          :gradient-normalization :none}})
          nn-conf-obj (eval nn-conf-code)

          ;; fine tune conf code and obj
          ft-conf-code (fine-tune-conf/new-fine-tune-conf :activation-fn :relu
                                                          :n-iterations 2
                                                          :regularization? false
                                                          :seed 1234)
          ft-conf-obj (eval-and-build ft-conf-code)

          ;; mln conf obj and code
          mln-conf-code (nn/builder :seed 123
                                    :optimization-algo :stochastic-gradient-descent
                                    :iterations 1
                                    :default-learning-rate 0.006
                                    :default-updater :nesterovs
                                    :default-momentum 0.9
                                    :regularization? false
                                    :build? true
                                    :default-gradient-normalization :none
                                    :backprop? true
                                    :pretrain? false
                                    :layers {0 {:dense-layer {:n-in 784
                                                              :n-out 1000
                                                              :layer-name "first layer"
                                                              :activation-fn :relu
                                                              :weight-init :xavier}}
                                             1 {:output-layer {:n-in 1000
                                                               :n-out 10
                                                               :layer-name "second layer"
                                                               :activation-fn :soft-max
                                                               :weight-init :xavier}}})
          mln-conf-obj (eval mln-conf-code)
          ;; initialized multi layer newtork
          mln (model/init! :model (multi-layer-network/new-multi-layer-network
                                   :conf mln-conf-code))

          mln-code (multi-layer-network/new-multi-layer-network
                    :conf mln-conf-code)

          helper (tl-helper/new-helper :mln mln :frozen-til 0 :as-code? false)
          featurized (featurize :helper helper
                                :data-set (get-example :ds (new-mnist-ds :as-code? false)
                                                       :idx 0))
          featurized-input (get-features featurized)
          tlb-code (tl/builder
                    :mln (multi-layer-network/new-multi-layer-network
                          :conf
                          (nn/builder
                           :seed 123
                           :optimization-algo :stochastic-gradient-descent
                           :iterations 1
                           :default-learning-rate 0.006
                           :default-updater :nesterovs
                           :default-momentum 0.9
                           :regularization? false
                           :build? true
                           :default-gradient-normalization :none
                           :backprop? true
                           :pretrain? false
                           :layers {0 {:dense-layer {:n-in 10
                                                     :n-out 100
                                                     :layer-name "first layer"
                                                     :activation-fn :relu
                                                     :weight-init :xavier}}
                                    1 {:activation-layer {:n-in 100
                                                          :n-out 10
                                                          :layer-name "second layer"
                                                          :activation-fn :soft-max
                                                          :weight-init :xavier}}
                                    2 {:output-layer {:n-in 10
                                                      :n-out 1
                                                      :layer-name "output layer"
                                                      :activation-fn :soft-max
                                                      :weight-init :xavier}}})))
          tlb-obj (eval-and-build tlb-code)]
      (testing "dl4clj.nn.transfer-learning.fine-tune-conf"
        (is (= org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
               (type ft-conf-obj)))
        (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
               (type (applied-to-nn-conf! :fine-tune-conf ft-conf-obj
                                          :nn-conf nn-conf-obj))))
        (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
               (type (nn-conf-from-fine-tune-conf ft-conf-obj
                                                  :build? false))))
        (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
               (type (nn-conf-from-fine-tune-conf ft-conf-obj
                                                  :build? true)))))
      (testing "dl4clj.nn.transfer-learning.helper"
        (is (= org.deeplearning4j.nn.transferlearning.TransferLearningHelper
               (type (tl-helper/new-helper :mln mln :as-code? false))))
        (is (= org.deeplearning4j.nn.transferlearning.TransferLearningHelper
               (type helper)))
        (is (= org.nd4j.linalg.dataset.DataSet (type featurized)))
        (is (= org.deeplearning4j.nn.transferlearning.TransferLearningHelper
               (type (fit-featurized! :helper helper :data-set featurized))))
        (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
               (type (output-from-featurized :helper helper :featurized-input featurized-input))))
        (is (= org.deeplearning4j.nn.multilayer.MultiLayerNetwork
               (type (unfrozen-mln helper)))))

      (testing "dl4clj.nn.transfer-learning.transfer-learning/builder helper fns"
        (is (= '[0 1001 (dl4clj.constants/value-of {:weight-init :xavier-uniform})]
               (tl/replace-layer-helper {:layer-idx 0 :n-out 1001 :weight-init :xavier-uniform})))
        (is (= '[0 1001 (org.deeplearning4j.nn.conf.distribution.NormalDistribution. 0 1)
                 (dl4clj.constants/value-of {:weight-init :xavier-uniform})]
               (tl/replace-layer-helper {:layer-idx 0 :n-out 1001
                                         :weight-init :xavier-uniform
                                         :dist {:normal {:mean 0 :std 1}}})))
        (is (= '[0 (org.deeplearning4j.nn.conf.preprocessor.UnitVarianceProcessor.)]
               (tl/input-pre-processor-helper {:layer-idx 0 :pre-processor (new-unit-variance-processor)})))
        (is (= '[0 (dl4clj.nn.conf.input-pre-processor/pre-processors {:unit-variance-processor {}})]
               (tl/input-pre-processor-helper {:layer-idx 0 :pre-processor {:unit-variance-processor {}}})))
        (is (= '[(dl4clj.utils/eval-and-build
                  (doto (org.deeplearning4j.nn.conf.layers.OutputLayer$Builder.)
                    (.nOut 10) (.activation (dl4clj.constants/value-of {:activation-fn :tanh}))
                    (.gradientNormalization (dl4clj.constants/value-of {:gradient-normalization :none}))
                    (.nIn 100) (.name "another layer")))]
               (tl/add-layers-helper (layer/output-layer-builder
                                      :n-in 100
                                      :n-out 10
                                      :layer-name "another layer"
                                      :activation-fn :tanh
                                      :gradient-normalization :none))))
        (is (= '[(dl4clj.utils/eval-and-build
                  (dl4clj.nn.conf.builders.layers/builder
                   {:output-layer
                    {:n-in 100, :n-out 10,
                     :layer-name "replacement second layer",
                     :activation-fn :soft-max, :weight-init :xavier}}))]
               (tl/add-layers-helper {:output-layer {:n-in 100
                                                     :n-out 10
                                                     :layer-name "replacement second layer"
                                                     :activation-fn :soft-max
                                                     :weight-init :xavier}})))
        (is (= '[[(dl4clj.utils/eval-and-build
                   (doto (org.deeplearning4j.nn.conf.layers.ActivationLayer$Builder.)
                     (.nOut 100) (.activation (dl4clj.constants/value-of {:activation-fn :tanh}))
                     (.gradientNormalization (dl4clj.constants/value-of {:gradient-normalization :none}))
                     (.nIn 10) (.name "replacement another layer")))]
                 [(dl4clj.utils/eval-and-build
                   (dl4clj.nn.conf.builders.layers/builder
                    {:output-layer
                     {:n-in 100, :n-out 10,
                      :layer-name "replacement second layer",
                      :activation-fn :soft-max, :weight-init :xavier}}))]]
               (tl/add-layers-helper {1 (layer/activation-layer-builder
                                         :n-in 10
                                         :n-out 100
                                         :layer-name "replacement another layer"
                                         :activation-fn :tanh
                                         :gradient-normalization :none)
                                      2 {:output-layer {:n-in 100
                                                        :n-out 10
                                                        :layer-name "replacement second layer"
                                                        :activation-fn :soft-max
                                                        :weight-init :xavier}}})))
        (is (= '(doto
                    (doto
                        (org.deeplearning4j.nn.transferlearning.TransferLearning$Builder.
                         (if
                             (dl4clj.nn.api.multi-layer-network/is-init-called?
                              (org.deeplearning4j.nn.multilayer.MultiLayerNetwork.
                               (.build
                                (doto
                                    (.list
                                     (doto
                                         (org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder.)))
                                  (.layer
                                   0
                                   (dl4clj.utils/eval-and-build
                                    (dl4clj.nn.conf.builders.layers/builder
                                     {:dense-layer
                                      {:n-in 100,
                                       :n-out 10,
                                       :layer-name "first layer",
                                       :activation-fn :tanh,
                                       :weight-init :relu}})))
                                  (.layer
                                   1
                                   (dl4clj.utils/eval-and-build
                                    (doto
                                        (org.deeplearning4j.nn.conf.layers.DenseLayer$Builder.)
                                      (.nOut 10)
                                      (.activation
                                       (dl4clj.constants/value-of {:activation-fn :tanh}))
                                      (.gradientNormalization
                                       (dl4clj.constants/value-of
                                        {:gradient-normalization :none}))
                                      (.nIn 10)
                                      (.name "second layer"))))))))
                           (org.deeplearning4j.nn.multilayer.MultiLayerNetwork.
                            (.build
                             (doto
                                 (.list
                                  (doto
                                      (org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder.)))
                               (.layer
                                0
                                (dl4clj.utils/eval-and-build
                                 (dl4clj.nn.conf.builders.layers/builder
                                  {:dense-layer
                                   {:n-in 100,
                                    :n-out 10,
                                    :layer-name "first layer",
                                    :activation-fn :tanh,
                                    :weight-init :relu}})))
                               (.layer
                                1
                                (dl4clj.utils/eval-and-build
                                 (doto
                                     (org.deeplearning4j.nn.conf.layers.DenseLayer$Builder.)
                                   (.nOut 10)
                                   (.activation
                                    (dl4clj.constants/value-of {:activation-fn :tanh}))
                                   (.gradientNormalization
                                    (dl4clj.constants/value-of {:gradient-normalization :none}))
                                   (.nIn 10)
                                   (.name "second layer")))))))
                           (dl4clj.nn.api.model/init!
                            :model
                            (org.deeplearning4j.nn.multilayer.MultiLayerNetwork.
                             (.build
                              (doto
                                  (.list
                                   (doto
                                       (org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder.)))
                                (.layer
                                 0
                                 (dl4clj.utils/eval-and-build
                                  (dl4clj.nn.conf.builders.layers/builder
                                   {:dense-layer
                                    {:n-in 100,
                                     :n-out 10,
                                     :layer-name "first layer",
                                     :activation-fn :tanh,
                                     :weight-init :relu}})))
                                (.layer
                                 1
                                 (dl4clj.utils/eval-and-build
                                  (doto
                                      (org.deeplearning4j.nn.conf.layers.DenseLayer$Builder.)
                                    (.nOut 10)
                                    (.activation
                                     (dl4clj.constants/value-of {:activation-fn :tanh}))
                                    (.gradientNormalization
                                     (dl4clj.constants/value-of
                                      {:gradient-normalization :none}))
                                    (.nIn 10)
                                    (.name "second layer"))))))))))
                      (.nOutReplace 0 101 (dl4clj.constants/value-of {:weight-init :relu}))
                      (.setInputPreProcessor
                       0
                       (dl4clj.nn.conf.input-pre-processor/pre-processors
                        {:unit-variance-processor {}}))
                      (.fineTuneConfiguration
                       (dl4clj.utils/eval-and-build
                        (doto
                            (org.deeplearning4j.nn.transferlearning.FineTuneConfiguration$Builder.)
                          (.iterations 2)
                          (.activation (dl4clj.constants/value-of {:activation-fn :relu}))
                          (.regularization false)
                          (.seed 1234))))
                      (.removeLayersFromOutput 1))
                  (.addLayer
                   (dl4clj.utils/eval-and-build
                    (dl4clj.nn.conf.builders.layers/builder
                     {:dense-layer
                      {:n-in 100,
                       :n-out 10,
                       :layer-name "third layer",
                       :activation-fn :tanh,
                       :weight-init :relu}})))
                  (.addLayer
                   (dl4clj.utils/eval-and-build
                    (dl4clj.nn.conf.builders.layers/builder
                     {:dense-layer
                      {:n-in 100,
                       :n-out 10,
                       :layer-name "4th layer",
                       :activation-fn :tanh,
                       :weight-init :relu}})))
                  (.addLayer
                   (dl4clj.utils/eval-and-build
                    (doto
                        (org.deeplearning4j.nn.conf.layers.OutputLayer$Builder.)
                      (.nOut 10)
                      (.activation (dl4clj.constants/value-of {:activation-fn :tanh}))
                      (.gradientNormalization
                       (dl4clj.constants/value-of {:gradient-normalization :none}))
                      (.nIn 10)
                      (.name "5th layer"))))
                  (.addLayer
                   (dl4clj.utils/eval-and-build
                    (dl4clj.nn.conf.builders.layers/builder
                     {:dense-layer
                      {:n-in 100,
                       :n-out 10,
                       :layer-name "last layer",
                       :activation-fn :tanh,
                       :weight-init :relu}}))))
               (tl/builder
                :mln
                (multi-layer-network/new-multi-layer-network
                 :as-code? true
                 :conf
                 (nn/builder :layers {0 {:dense-layer {:n-in 100
                                                       :n-out 10
                                                       :layer-name "first layer"
                                                       :activation-fn :tanh
                                                       :weight-init :relu}}
                                      1 (dl4clj.nn.conf.builders.layers/dense-layer-builder
                                         :n-in 10
                                         :n-out 10
                                         :layer-name "second layer"
                                         :activation-fn :tanh
                                         :gradient-normalization :none)}))
                :fine-tune-conf (fine-tune-conf/new-fine-tune-conf
                                 :activation-fn :relu
                                 :n-iterations 2
                                 :regularization? false
                                 :seed 1234)
                :remove-last-n-layers 1
                :replacement-layer {:layer-idx 0 :n-out 101 :weight-init :relu}
                :add-layers {2 {:dense-layer {:n-in 100
                                              :n-out 10
                                              :layer-name "third layer"
                                              :activation-fn :tanh
                                              :weight-init :relu}}
                             4 (layer/output-layer-builder
                                :n-in 10
                                :n-out 10
                                :layer-name "5th layer"
                                :activation-fn :tanh
                                :gradient-normalization :none)
                             5 {:dense-layer {:n-in 100
                                              :n-out 10
                                              :layer-name "last layer"
                                              :activation-fn :tanh
                                              :weight-init :relu}}
                             3 {:dense-layer {:n-in 100
                                              :n-out 10
                                              :layer-name "4th layer"
                                              :activation-fn :tanh
                                              :weight-init :relu}}}
                :as-code? true
                :input-pre-processor {:layer-idx 0
                                      :pre-processor {:unit-variance-processor {}}}))))
      (testing "dl4clj.nn.transfer-learning.transfer-learning"
        (is (= org.deeplearning4j.nn.multilayer.MultiLayerNetwork
               (type (tl/builder
                      :mln mln-code
                      :fine-tune-conf ft-conf-code
                      :remove-output-layer? true
                      :replacement-layer {:layer-idx 0 :n-out 1001 :weight-init :xavier-uniform}
                      :remove-last-n-layers 1
                      :add-layers (layer/output-layer-builder
                                   :n-in 100
                                   :n-out 10
                                   :layer-name "another layer"
                                   :activation-fn :tanh
                                   :gradient-normalization :none)
                      :set-feature-extractor-idx 0
                      :input-pre-processor {:layer-idx 0 :pre-processor (new-unit-variance-processor)}
                      :as-code? false))))
        (is (= ["first layer" "replacement another layer" "replacement second layer"]
               (mln/get-layer-names
                (tl/builder :tlb tlb-code
                            :fine-tune-conf ft-conf-code
                            :remove-last-n-layers 2
                            :add-layers {1 (layer/activation-layer-builder
                                            :n-in 10
                                            :n-out 100
                                            :layer-name "replacement another layer"
                                            :activation-fn :tanh
                                            :gradient-normalization :none)
                                         2 {:output-layer {:n-in 100
                                                           :n-out 10
                                                           :layer-name "replacement second layer"
                                                           :activation-fn :soft-max
                                                           :weight-init :xavier}}}
                            :as-code? false))))
        (is (= ["first layer" "another layer"]
               (mln/get-layer-names
                (tl/builder
                 :mln mln-code
                 :fine-tune-conf ft-conf-code
                 :remove-output-layer? true
                 :add-layers (layer/output-layer-builder
                              :n-in 100
                              :n-out 10
                              :layer-name "another layer"
                              :activation-fn :tanh
                              :gradient-normalization :none)
                 :set-feature-extractor-idx 0
                 :input-pre-processor {:layer-idx 0 :pre-processor (new-unit-variance-processor)}
                 :as-code? false))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; any layer builder
;; dl4clj.nn.conf.builders.builders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest activation-layer-test
  (testing "the creation of a activation layer from a nn-conf"
    (let [conf {:activation-layer
                {:n-in 10 :n-out 2 :activation-fn :relu
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo" :learning-rate 0.1
                 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :momentum 0.2 :momentum-after {0 0.3 1 0.4}
                 :updater :nesterovs :weight-init :distribution}}]
      (is (= :activation (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.ActivationLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest center-loss-output-layer-test
  (testing "the creation of a ceneter loss output layer from a nn-conf"
    (let [conf {:center-loss-output-layer
                {:alpha 0.1 :gradient-check? false :lambda 0.1
                 :loss-fn :mse :layer-name "foo1"
                 :activation-fn :relu
                 :adam-mean-decay 0.2 :adam-var-decay 0.1
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :updater :adam
                 :weight-init :distribution}}]
      (is (= :center-loss-output-layer (layer-creation/layer-type
                                        {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.training.CenterLossOutputLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest output-layer-test
  (testing "the creation of a output layer from a nn-conf"
    (let [conf {:output-layer
                {:n-in 10 :n-out 2 :loss-fn :mse
                 :activation-fn :relu
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo2"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :rho 0.7 :updater :adadelta
                 :weight-init :distribution}}]
      (is (= :output (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.OutputLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest rnn-output-layer-test
  (testing "the creation of a rnn output layer from a nn-conf"
    (let [conf {:rnn-output-layer
                {:n-in 10 :n-out 2 :loss-fn :mse
                 :activation-fn :relu
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo3"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :rms-decay 0.7 :updater :rmsprop
                 :weight-init :distribution}}]
      (is (= :rnnoutput (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest autoencoder-layer-test
  (testing "the creation of a autoencoder layer from a nn-conf"
    (let [conf {:auto-encoder
                {:n-in 10 :n-out 2 :pre-train-iterations 2
                 :loss-fn :mse :visible-bias-init 0.1
                 :corruption-level 0.7 :sparsity 0.4
                 :activation-fn :relu
                 :adam-mean-decay 0.2 :adam-var-decay 0.1
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo4"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :updater :adam
                 :weight-init :distribution}}]
      (is (= :auto-encoder (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest rbm-layer-test
  (testing "the creation of a rbm layer from a nn-conf"
    (let [conf {:rbm {:n-in 10 :n-out 2 :loss-fn :mse
                      :pre-train-iterations 1 :visible-bias-init 0.7
                      :hidden-unit :softmax :visible-unit :identity
                      :k 2 :sparsity 0.6
                      :activation-fn :relu
                      :adam-mean-decay 0.2 :adam-var-decay 0.1
                      :bias-init 0.7 :bias-learning-rate 0.1
                      :dist {:normal {:mean 0 :std 1}}
                      :drop-out 0.2 :epsilon 0.3
                      :gradient-normalization :none
                      :gradient-normalization-threshold 0.9
                      :layer-name "foo5"
                      :learning-rate 0.1 :learning-rate-policy :inverse
                      :learning-rate-schedule {0 0.2 1 0.5}
                      :updater :adam :weight-init :distribution}}]
      (is (= :rbm (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.feedforward.rbm.RBM
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest graves-bidirectional-lstm-layer-test
  (testing "the creation of a bidirectional lstm layer from a nn-conf"
    (let [conf {:graves-bidirectional-lstm
                {:n-in 10 :n-out 2 :forget-gate-bias-init 0.2
                 :gate-activation-fn :relu
                 :activation-fn :relu
                 :adam-mean-decay 0.2 :adam-var-decay 0.1
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo6"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :updater :adam :weight-init :distribution}}]
      (is (= :graves-bidirectional-lstm (layer-creation/layer-type
                                         {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.recurrent.GravesBidirectionalLSTM
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest graves-lstm-layer-test
  (testing "the creation of a lstm layer from a nn-conf"
    (let [conf {:graves-lstm
                {:n-in 10 :n-out 2 :forget-gate-bias-init 0.2
                 :gate-activation-fn :relu
                 :activation-fn :relu
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo7"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :momentum 0.2 :momentum-after {0 0.3 1 0.4}
                 :updater :nesterovs :weight-init :distribution}}]
      (is (= :graves-lstm (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.recurrent.GravesLSTM
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest batch-normalization-layer-test
  (testing "the creation of a batch normalization layer from a nn-conf"
    (let [conf {:batch-normalization
                {:n-in 10 :n-out 2 :beta 0.5
                 :decay 0.3 :eps 0.1 :gamma 0.1
                 :mini-batch? false :lock-gamma-beta? true
                 :activation-fn :relu
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo8"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :rho 0.7 :updater :adadelta
                 :weight-init :distribution}}]
      (is (= :batch-normalization (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.normalization.BatchNormalization
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest convolution-layer-test
  (testing "the creation of a convolution layer from a nn-conf"
    (let [conf {:convolutional-layer
                {:n-in 10 :n-out 2
                 :kernel-size [2 2] :padding [2 2] :stride [2 2]
                 :activation-fn :relu
                 :adam-mean-decay 0.2 :adam-var-decay 0.1
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo9"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :updater :adam :weight-init :distribution}}
          conf-1d {:convolution-1d-layer
                   {:n-in 10 :n-out 2
                    :kernel-size 6 :stride 3 :padding 3
                    :activation-fn :relu
                    :bias-init 0.7 :bias-learning-rate 0.1
                    :dist {:normal {:mean 0 :std 1}}
                    :drop-out 0.2 :epsilon 0.3
                    :gradient-normalization :none
                    :gradient-normalization-threshold 0.9
                    :layer-name "foo10"
                    :learning-rate 0.1 :learning-rate-policy :inverse
                    :learning-rate-schedule {0 0.2 1 0.5}
                    :rms-decay 0.7 :updater :rmsprop
                    :weight-init :distribution}}]
      (is (= :convolution (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
       (is (= org.deeplearning4j.nn.layers.convolution.ConvolutionLayer
              (type (layer-creation/new-layer
                     :as-code? false
                     :nn-conf (quick-nn-conf conf)))))

      (is (= :convolution1d (layer-creation/layer-type {:nn-conf (quick-nn-conf conf-1d)})))
      (is (= org.deeplearning4j.nn.layers.convolution.Convolution1DLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf-1d))))))))

(deftest dense-layer-test
  (testing "creation of a dense layer from a nn-conf"
    (let [conf {:dense-layer {:n-in 10 :n-out 2
                              :activation-fn :relu
                              :adam-mean-decay 0.2 :adam-var-decay 0.1
                              :bias-init 0.7 :bias-learning-rate 0.1
                              :dist {:normal {:mean 0 :std 1}}
                              :drop-out 0.2 :epsilon 0.3
                              :gradient-normalization :none
                              :gradient-normalization-threshold 0.9
                              :layer-name "foo11"
                              :learning-rate 0.1 :learning-rate-policy :inverse
                              :learning-rate-schedule {0 0.2 1 0.5}
                              :updater :adam :weight-init :distribution}}]
      (is (= :dense (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest embedding-layer-test
  (testing "the creation of a embedding layer from a nn-conf"
    (let [conf {:embedding-layer
                {:n-in 10 :n-out 2
                 :activation-fn :relu
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo12"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :rms-decay 0.7 :updater :rmsprop
                 :weight-init :distribution}}]
      (is (= :embedding (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest local-response-normalization-layer-test
  (testing "the creation of a local response normalization layer from a nn-conf"
    (let [conf {:local-response-normalization
                {:alpha 0.2 :beta 0.2 :k 0.2 :n 1
                 :activation-fn :relu
                 :adam-mean-decay 0.2 :adam-var-decay 0.1
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo13"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :updater :adam :weight-init :distribution}}]
      (is (= :local-response-normalization (layer-creation/layer-type
                                            {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest subsampling-layer-test
  (testing "the creation of a subsampling layer from a nn-conf"
    (let [conf {:subsampling-layer
                {:kernel-size [2 2] :stride [2 2] :padding [2 2]
                 :pooling-type :sum
                 :build? true
                 :activation-fn :relu
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo14"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :momentum 0.2 :momentum-after {0 0.3 1 0.4}
                 :updater :nesterovs :weight-init :distribution}}
          conf-1d {:subsampling-1d-layer
                   {:kernel-size 2 :stride 2 :padding 2
                    :pooling-type :sum
                    :build? true
                    :activation-fn :relu
                    :adam-mean-decay 0.2 :adam-var-decay 0.1
                    :bias-init 0.7 :bias-learning-rate 0.1
                    :dist {:normal {:mean 0 :std 1}}
                    :drop-out 0.2 :epsilon 0.3
                    :gradient-normalization :none
                    :gradient-normalization-threshold 0.9
                    :layer-name "foo15"
                    :learning-rate 0.1 :learning-rate-policy :inverse
                    :learning-rate-schedule {0 0.2 1 0.5}
                    :updater :adam :weight-init :distribution}}]

      (is (= :subsampling (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf)))))
      (is (= :subsampling1d (layer-creation/layer-type {:nn-conf (quick-nn-conf conf-1d)})))
      (is (= org.deeplearning4j.nn.layers.convolution.subsampling.Subsampling1DLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf-1d))))))))

(deftest loss-layer-test
  (testing "the creation of a loss layer from a nn-conf"
    (let [conf {:loss-layer {:loss-fn :mse
                             :activation-fn :relu
                             :adam-mean-decay 0.2 :adam-var-decay 0.1
                             :bias-init 0.7 :bias-learning-rate 0.1
                             :dist {:normal {:mean 0 :std 1}}
                             :drop-out 0.2 :epsilon 0.3
                             :gradient-normalization :none
                             :gradient-normalization-threshold 0.9
                             :layer-name "foo16"
                             :learning-rate 0.1 :learning-rate-policy :inverse
                             :learning-rate-schedule {0 0.2 1 0.5}
                             :updater :adam :weight-init :distribution}}]
      (is (= :loss (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.LossLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest dropout-layer-test
  (testing "the creation of a dropout layer from a nn-conf"
    (let [conf {:dropout-layer
                {:n-in 2 :n-out 10
                 :activation-fn :relu
                 :adam-mean-decay 0.2 :adam-var-decay 0.1
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo17"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :updater :adam :weight-init :distribution}}]
      (is (= :dropout (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.DropoutLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest global-pooling-layer-test
  (testing "the creation of a global pooling layer from a nn-conf, also shows off layer validation
 \n this triggers warnings because you can't set the updater so the values for updaters are automatically set to nil"
     (let [conf {:global-pooling-layer
                {:pooling-dimensions [3 2]
                 :collapse-dimensions? true
                 :pnorm 2
                 :pooling-type :pnorm
                 :activation-fn :relu
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo18"
                 :updater :rmsprop
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :weight-init :distribution}}]
      (is (= org.deeplearning4j.nn.layers.pooling.GlobalPoolingLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest zero-padding-layer-test
  (testing "the creation of a zero padding layer from a nn-conf"
    (let [conf {:zero-padding-layer
                {:pad-top 1 :pad-bot 2 :pad-left 3 :pad-right 4
                 :activation-fn :relu
                 :adam-mean-decay 0.2 :adam-var-decay 0.1
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo19"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :updater :adam :weight-init :distribution}}]
      (is (= :zero-padding (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.convolution.ZeroPaddingLayer
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest variational-autoencoder-layer-test
  (testing "the creation of a vae layer from a nn-conf"
    (let [conf {:variational-auto-encoder
                {:n-in 5 :n-out 10 :loss-fn :mse
                 :pre-train-iterations 1 :visible-bias-init 2
                 :decoder-layer-sizes [5 9]
                 :encoder-layer-sizes [7 2]
                 :reconstruction-distribution {:composite
                                               {:distributions
                                                [{:gaussian-reconstruction
                                                  {:activation-fn :tanh
                                                   :dist-size 5}}
                                                 {:bernoulli {:dist-size 1}}]}}
                 :vae-loss-fn {:output-activation-fn :sigmoid :loss-fn :mse}
                 :num-samples 2 :pzx-activation-function :tanh
                 :activation-fn :relu
                 :adam-mean-decay 0.2 :adam-var-decay 0.1
                 :bias-init 0.7 :bias-learning-rate 0.1
                 :dist {:normal {:mean 0 :std 1}}
                 :drop-out 0.2 :epsilon 0.3
                 :gradient-normalization :none
                 :gradient-normalization-threshold 0.9
                 :layer-name "foo20"
                 :learning-rate 0.1 :learning-rate-policy :inverse
                 :learning-rate-schedule {0 0.2 1 0.5}
                 :updater :adam :weight-init :distribution}}]
      (is (= :variational-autoencoder (layer-creation/layer-type {:nn-conf (quick-nn-conf conf)})))
      (is (= org.deeplearning4j.nn.layers.variational.VariationalAutoencoder
             (type (layer-creation/new-layer
                    :as-code? false
                    :nn-conf (quick-nn-conf conf))))))))

(deftest frozen-layer-test
  (testing "the creation of a frozen layer from an existing layer"
    (let [layer-conf {:dense-layer
                      {:n-in 10 :n-out 2
                       :activation-fn :relu
                       :adam-mean-decay 0.2 :adam-var-decay 0.1
                       :bias-init 0.7 :bias-learning-rate 0.1
                       :dist {:normal {:mean 0 :std 1}}
                       :drop-out 0.2 :epsilon 0.3
                       :gradient-normalization :none
                       :gradient-normalization-threshold 0.9
                       :layer-name "foo11"
                       :learning-rate 0.1 :learning-rate-policy :inverse
                       :learning-rate-schedule {0 0.2 1 0.5}
                       :updater :adam :weight-init :distribution}}]
      (is (= org.deeplearning4j.nn.layers.FrozenLayer
             (type (layer-creation/new-frozen-layer
                    :as-code? false
                    :layer
                    (model/set-param-table!
                     :model (layer-creation/new-layer
                             :nn-conf (quick-nn-conf layer-conf))
                     :param-table-map {"foo" (indarray-of-zeros :rows 1
                                                                :as-code? true)}))))))))
