(ns dl4clj.nn-tests
  (:require [dl4clj.nn.conf.builders.builders :refer :all]
            [dl4clj.nn.conf.builders.nn-conf-builder :refer :all]
            [dl4clj.nn.conf.builders.multi-layer-builders :refer :all]
            [dl4clj.nn.api.multi-layer-network :refer :all]
            [dl4clj.nn.multilayer.multi-layer-network :refer :all]
            [dl4clj.nn.conf.distributions :refer :all]
            [dl4clj.nn.conf.variational.dist-builders :refer :all]
            [dl4clj.nn.conf.step-fns :refer :all]
            [dl4clj.constants :refer :all]
            [dl4clj.nn.gradient.default-gradient :refer :all]
            [dl4clj.nn.params.param-initializers :refer :all]
            [dl4clj.nn.layers.layer-creation :refer :all]
            [dl4clj.nn.api.layer :refer :all]
            [dl4clj.nn.api.layer-specific-fns :refer :all]
            [dl4clj.nn.api.model :refer :all]
            [dl4clj.nn.conf.input-pre-processor :refer :all]
            [dl4clj.nn.transfer-learning.fine-tune-conf :refer :all]
            [dl4clj.nn.transfer-learning.helper :refer :all]
            [dl4clj.nn.transfer-learning.transfer-learning :refer :all]
            [dl4clj.nn.updater.layer-updater :refer :all]
            [dl4clj.nn.updater.multi-layer-updater :refer :all]
            [dl4clj.nn.api.nn-conf :refer :all]
            ;; helper fns
            [dl4clj.utils :refer [array-of get-labels]]
            [nd4clj.linalg.factory.nd4j :refer [indarray-of-zeros]]
            [dl4clj.datasets.default-datasets :refer [new-mnist-ds]]
            [dl4clj.datasets.iterators :refer [new-mnist-data-set-iterator]]
            [dl4clj.datasets.api.datasets :refer [get-features get-example]]
            [dl4clj.eval.evaluation :refer [new-classification-evaler]]
            [dl4clj.eval.api.eval :refer [eval-classification! get-stats
                                          eval-model-whole-ds]]
            [clojure.test :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; helper fn for layer creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn quick-nn-conf
  [layer]
  (nn-conf-builder :optimization-algo :stochastic-gradient-descent
                   :iterations 1
                   :learning-rate 0.006
                   :lr-policy-decay-rate 0.2
                   :build? true
                   :layer layer))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; distributions to sample weights from
;; dl4clj.nn.conf.distribution.distribution
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest distributions-test
  (testing "the creation of distributions for use in a nn-conf"
    (is (= org.deeplearning4j.nn.conf.distribution.UniformDistribution
           (type (new-uniform-distribution :lower 0.2 :upper 0.4))))
    (is (= org.deeplearning4j.nn.conf.distribution.NormalDistribution
           (type (new-normal-distribution :mean 0 :std 1))))
    (is (= org.deeplearning4j.nn.conf.distribution.GaussianDistribution
           (type (new-gaussian-distribution :mean 0.0 :std 1))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; reconstruction distribution
;; dl4clj.nn.conf.variational.dist-builders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest reconstruction-distribution-test
  (testing "the creation of reconstruction distributions for vae's"
    (is (= org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution
           (type (new-bernoulli-reconstruction-distribution :activation-fn :sigmoid))))
    (is (= org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution
           (type (new-bernoulli-reconstruction-distribution))))

    (is (= org.deeplearning4j.nn.conf.layers.variational.ExponentialReconstructionDistribution
           (type (new-exponential-reconstruction-distribution :activation-fn :relu))))
    (is (= org.deeplearning4j.nn.conf.layers.variational.ExponentialReconstructionDistribution
           (type (new-exponential-reconstruction-distribution))))

    (is (= org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution
           (type (new-gaussian-reconstruction-distribution :activation-fn :relu))))
    (is (= org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution
           (type (new-gaussian-reconstruction-distribution))))

    (is (= org.deeplearning4j.nn.conf.layers.variational.CompositeReconstructionDistribution
           (type (new-composite-reconstruction-distribution
                  ;; using a user facing fn
                  {0 {:dist (new-bernoulli-reconstruction-distribution :activation-fn :sigmoid)
                      :dist-size 2}
                   ;; using the multi method
                   1 {:bernoulli {:activation-fn :sigmoid
                                  :dist-size 5}}

                   2 {:exponential {:activation-fn :sigmoid
                                    :dist-size 3}}

                   3 {:gaussian {:activation-fn :hard-tanh
                                 :dist-size 1}}
                   4 {:bernoulli {:activation-fn :sigmoid
                                  :dist-size 4}}
                   ;; explicitly using the multimethod
                   5 {:dist (distributions {:bernoulli {:activation-fn :sigmoid}})
                      :dist-size 7}}))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; step functions for use in nn-conf creation
;; dl4clj.nn.conf.step-fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest step-fn-test
  (testing "the creation of step fns"
    (is (= org.deeplearning4j.nn.conf.stepfunctions.DefaultStepFunction
           (type (new-default-step-fn))))
    (is (= org.deeplearning4j.nn.conf.stepfunctions.GradientStepFunction
           (type (new-gradient-step-fn))))
    (is (= org.deeplearning4j.nn.conf.stepfunctions.NegativeDefaultStepFunction
           (type (new-negative-default-step-fn))))
    (is (= org.deeplearning4j.nn.conf.stepfunctions.NegativeGradientStepFunction
           (type (new-negative-gradient-step-fn))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; creation of default gradients
;; dl4clj.nn.gradient.default-gradient
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest default-gradient-test
  (testing "the creation and manipulation of gradients"
    (let [grad-with-var (set-gradient-for! :grad (new-default-gradient)
                                           :variable "foo"
                                           :new-gradient (indarray-of-zeros
                                                          :rows 2 :columns 2))]
     (is (= org.deeplearning4j.nn.gradient.DefaultGradient
           (type (new-default-gradient))))
    (is (= org.deeplearning4j.nn.gradient.DefaultGradient
           (type grad-with-var)))
    ;; I don't think this test is reliable bc it assumes cpu
    (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
           (type (gradient :grad grad-with-var))))
    (is (= java.util.LinkedHashMap
           (type (gradient-for-variable grad-with-var))))
    ;; gradient order was not explictly set
    (is (= nil
           (type (flattening-order-for-variables :grad grad-with-var
                                                 :variable "foo")))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; param initializers
;; dl4clj.nn.params.param-initializers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest param-initializers-test
  (testing "the creation of param initializers"
    (is (= org.deeplearning4j.nn.params.BatchNormalizationParamInitializer
           (type (new-batch-norm-initializer))))
    (is (= org.deeplearning4j.nn.params.CenterLossParamInitializer
           (type (new-center-loss-initializer))))
    (is (= org.deeplearning4j.nn.params.ConvolutionParamInitializer
           (type (new-convolution-initializer))))
    (is (= org.deeplearning4j.nn.params.DefaultParamInitializer
           (type (new-default-initializer))))
    (is (= org.deeplearning4j.nn.params.EmptyParamInitializer
           (type (new-empty-initializer))))
    (is (= org.deeplearning4j.nn.params.GravesBidirectionalLSTMParamInitializer
           (type (new-bidirectional-lstm-initializer))))
    (is (= org.deeplearning4j.nn.params.GravesLSTMParamInitializer
           (type (new-lstm-initializer))))
    (is (= org.deeplearning4j.nn.params.PretrainParamInitializer
           (type (new-pre-train-initializer))))
    (is (= org.deeplearning4j.nn.params.VariationalAutoencoderParamInitializer
           (type (new-vae-initializer))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; any layer builder
;; dl4clj.nn.conf.builders.builders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest layer-builder-test
  (testing "the creation of nearly any layer in dl4j"
    (let [activation-layer-conf (activation-layer-builder
                                 :n-in 10 :n-out 2 :activation-fn :relu
                                 :bias-init 0.7 :bias-learning-rate 0.1
                                 :dist (new-normal-distribution :mean 0 :std 1)
                                 :drop-out 0.2 :epsilon 0.3
                                 :gradient-normalization :none
                                 :gradient-normalization-threshold 0.9
                                 :layer-name "foo" :learning-rate 0.1
                                 :learning-rate-policy :inverse
                                 :learning-rate-schedule {0 0.2 1 0.5}
                                 :momentum 0.2 :momentum-after {0 0.3 1 0.4}
                                 :updater :nesterovs :weight-init :distribution)
          center-loss-output-layer-conf (center-loss-output-layer-builder
                                         :alpha 0.1 :gradient-check? false :lambda 0.1
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
                                         :weight-init :distribution)
          output-layer-conf (output-layer-builder
                             :n-in 10 :n-out 2 :loss-fn :mse
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
                             :weight-init :distribution)
          rnn-output-layer-conf (rnn-output-layer-builder
                                 :n-in 10 :n-out 2 :loss-fn :mse
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
                                 :weight-init :distribution)
          autoencoder-layer-conf (auto-encoder-layer-builder
                                  :n-in 10 :n-out 2 :pre-train-iterations 2
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
                                  :weight-init :distribution)
          rbm-layer-conf (rbm-layer-builder
                          :n-in 10 :n-out 2 :loss-fn :mse
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
                          :updater :adam :weight-init :distribution)
          graves-bidirectional-lstm-conf (graves-bidirectional-lstm-layer-builder
                                          :n-in 10 :n-out 2 :forget-gate-bias-init 0.2
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
                                          :updater :adam :weight-init :distribution)
          graves-lstm-layer-conf (graves-lstm-layer-builder
                                  :n-in 10 :n-out 2 :forget-gate-bias-init 0.2
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
                                  :updater :nesterovs :weight-init :distribution)
          batch-normalization-layer-conf (batch-normalization-layer-builder
                                          :n-in 10 :n-out 2 :beta 0.5
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
                                          :weight-init :distribution)
          convolutional-layer-conf (convolutional-layer-builder
                                    :n-in 10 :n-out 2
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
                                    :updater :adam :weight-init :distribution)
          convolutional-1d-layer-conf (convolution-1d-layer-builder
                                       :n-in 10 :n-out 2
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
                                       :weight-init :distribution)
          dense-layer-conf (dense-layer-builder
                            :n-in 10 :n-out 2
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
                            :updater :adam :weight-init :distribution)
          embedding-layer-conf (embedding-layer-builder
                                :n-in 10 :n-out 2
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
                                :weight-init :distribution)
          local-response-normalization-conf (local-response-normalization-layer-builder
                                             :alpha 0.2 :beta 0.2 :k 0.2 :n 1
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
                                             :updater :adam :weight-init :distribution)
          subsampling-layer-conf (subsampling-layer-builder
                                  :kernel-size [2 2] :stride [2 2] :padding [2 2]
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
                                  :updater :nesterovs :weight-init :distribution)
          subsampling-1d-layer-conf (subsampling-1d-layer-builder
                                     :kernel-size 2 :stride 2 :padding 2
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
                                     :updater :adam :weight-init :distribution)
          loss-layer-conf (loss-layer-builder
                           :loss-fn :mse
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
                           :updater :adam :weight-init :distribution)
          dropout-layer-conf (dropout-layer-builder
                              :n-in 2 :n-out 10
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
                              :updater :adam :weight-init :distribution)
          global-pooling-layer-conf (global-pooling-layer-builder
                                     :pooling-dimensions [3 2]
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
                                     :updater :none
                                     :learning-rate 0.1 :learning-rate-policy :inverse
                                     :learning-rate-schedule {0 0.2 1 0.5}
                                     :weight-init :distribution)
          zero-padding-layer-conf (zero-padding-layer-builder
                                   :pad-top 1 :pad-bot 2 :pad-left 3 :pad-right 4
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
                                   :updater :adam :weight-init :distribution)
          vae-layer-conf (variational-autoencoder-builder
                          :n-in 5 :n-out 10 :loss-fn :mse
                          :pre-train-iterations 1 :visible-bias-init 2
                          :decoder-layer-sizes [5 9]
                          :encoder-layer-sizes [7 2]
                          :reconstruction-distribution {:gaussian {:activation-fn :tanh}}
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
                          :updater :adam :weight-init :distribution)]
      ;; activation layer
      (is (= org.deeplearning4j.nn.conf.layers.ActivationLayer
             (type activation-layer-conf)))
      (is (= :activation (layer-type {:nn-conf (quick-nn-conf activation-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.ActivationLayer
             (type (new-layer :nn-conf (quick-nn-conf activation-layer-conf)))))

      ;; center loss layer
      (is (= org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer
             (type center-loss-output-layer-conf)))
      (is (= :center-loss-output-layer (layer-type {:nn-conf (quick-nn-conf center-loss-output-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.training.CenterLossOutputLayer
             (type (new-layer :nn-conf (quick-nn-conf center-loss-output-layer-conf)))))

      ;; output layer
      (is (= org.deeplearning4j.nn.conf.layers.OutputLayer
             (type output-layer-conf)))
      (is (= :output (layer-type {:nn-conf (quick-nn-conf output-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.OutputLayer
             (type (new-layer :nn-conf (quick-nn-conf output-layer-conf)))))

      ;; rnn output layer
      (is (= org.deeplearning4j.nn.conf.layers.RnnOutputLayer
             (type rnn-output-layer-conf)))
      (is (= :rnnoutput (layer-type {:nn-conf (quick-nn-conf rnn-output-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer
             (type (new-layer :nn-conf (quick-nn-conf rnn-output-layer-conf)))))

      ;; atuoencoders
      (is (= org.deeplearning4j.nn.conf.layers.AutoEncoder
             (type autoencoder-layer-conf)))
      (is (= :auto-encoder (layer-type {:nn-conf (quick-nn-conf autoencoder-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder
             (type (new-layer :nn-conf (quick-nn-conf autoencoder-layer-conf)))))

      ;; rbm
      (is (= org.deeplearning4j.nn.conf.layers.RBM (type rbm-layer-conf)))
      (is (= :rbm (layer-type {:nn-conf (quick-nn-conf rbm-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.feedforward.rbm.RBM
             (type (new-layer :nn-conf (quick-nn-conf rbm-layer-conf)))))

      ;; graves bidirectional lstm
      (is (= org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM
             (type graves-bidirectional-lstm-conf)))
      (is (= :graves-bidirectional-lstm (layer-type {:nn-conf (quick-nn-conf graves-bidirectional-lstm-conf)})))
      (is (= org.deeplearning4j.nn.layers.recurrent.GravesBidirectionalLSTM
             (type (new-layer :nn-conf (quick-nn-conf graves-bidirectional-lstm-conf)))))

      ;; graves lstm
      (is (= org.deeplearning4j.nn.conf.layers.GravesLSTM
             (type graves-lstm-layer-conf)))
      (is (= :graves-lstm (layer-type {:nn-conf (quick-nn-conf graves-lstm-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.recurrent.GravesLSTM
             (type (new-layer :nn-conf (quick-nn-conf graves-lstm-layer-conf)))))

      ;; batch normalization
      (is (= org.deeplearning4j.nn.conf.layers.BatchNormalization
             (type batch-normalization-layer-conf)))
      (is (= :batch-normalization (layer-type {:nn-conf (quick-nn-conf batch-normalization-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.normalization.BatchNormalization
             (type (new-layer :nn-conf (quick-nn-conf batch-normalization-layer-conf)))))

      ;; convolution
      (is (= org.deeplearning4j.nn.conf.layers.ConvolutionLayer
             (type convolutional-layer-conf)))
      (is (= :convolution (layer-type {:nn-conf (quick-nn-conf convolutional-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.convolution.ConvolutionLayer
             (type (new-layer :nn-conf (quick-nn-conf convolutional-layer-conf)))))

      ;; convolution1d
      (is (= org.deeplearning4j.nn.conf.layers.Convolution1DLayer
             (type convolutional-1d-layer-conf)))
      (is (= :convolution1d (layer-type {:nn-conf (quick-nn-conf convolutional-1d-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.convolution.Convolution1DLayer
             (type (new-layer :nn-conf (quick-nn-conf convolutional-1d-layer-conf)))))

      ;; dense
      (is (= org.deeplearning4j.nn.conf.layers.DenseLayer
             (type dense-layer-conf)))
      (is (= :dense (layer-type {:nn-conf (quick-nn-conf dense-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer
             (type (new-layer :nn-conf (quick-nn-conf dense-layer-conf)))))

      ;; embedding
      (is (= org.deeplearning4j.nn.conf.layers.EmbeddingLayer
             (type embedding-layer-conf)))
      (is (= :embedding (layer-type {:nn-conf (quick-nn-conf embedding-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.feedforward.embedding.EmbeddingLayer
             (type (new-layer :nn-conf (quick-nn-conf embedding-layer-conf)))))

      ;; local response normalization
      (is (= org.deeplearning4j.nn.conf.layers.LocalResponseNormalization
             (type local-response-normalization-conf)))
      (is (= :local-response-normalization (layer-type {:nn-conf (quick-nn-conf local-response-normalization-conf)})))
      (is (= org.deeplearning4j.nn.layers.normalization.LocalResponseNormalization
             (type (new-layer :nn-conf (quick-nn-conf local-response-normalization-conf)))))

      ;; subsampling
      (is (= org.deeplearning4j.nn.conf.layers.SubsamplingLayer
             (type subsampling-layer-conf)))
      (is (= :subsampling (layer-type {:nn-conf (quick-nn-conf subsampling-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer
             (type (new-layer :nn-conf (quick-nn-conf subsampling-layer-conf)))))

      ;; subsampling1d
      (is (= org.deeplearning4j.nn.conf.layers.Subsampling1DLayer
             (type subsampling-1d-layer-conf)))
      (is (= :subsampling1d (layer-type {:nn-conf (quick-nn-conf subsampling-1d-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.convolution.subsampling.Subsampling1DLayer
             (type (new-layer :nn-conf (quick-nn-conf subsampling-1d-layer-conf)))))

      ;; loss layer
      (is (= org.deeplearning4j.nn.conf.layers.LossLayer
             (type loss-layer-conf)))
      (is (= :loss (layer-type {:nn-conf (quick-nn-conf loss-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.LossLayer
             (type (new-layer :nn-conf (quick-nn-conf loss-layer-conf)))))

      ;; dropout
      (is (= org.deeplearning4j.nn.conf.layers.DropoutLayer
             (type dropout-layer-conf)))
      (is (= :dropout (layer-type {:nn-conf (quick-nn-conf dropout-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.DropoutLayer
             (type (new-layer :nn-conf (quick-nn-conf dropout-layer-conf)))))

      ;; global pooling
      ;; this triggers warnings because you can't set the updater
      ;; the set valuess are set to nil
      (is (= org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer
             (type global-pooling-layer-conf)))
      (is (= :global-pooling (layer-type {:nn-conf (quick-nn-conf global-pooling-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.pooling.GlobalPoolingLayer
             (type (new-layer :nn-conf (quick-nn-conf global-pooling-layer-conf)))))

      ;; zero padding
      (is (= org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer
             (type zero-padding-layer-conf)))
      (is (= :zero-padding (layer-type {:nn-conf (quick-nn-conf zero-padding-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.convolution.ZeroPaddingLayer
             (type (new-layer :nn-conf (quick-nn-conf zero-padding-layer-conf)))))

      ;; vae
      (is (= org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder
             (type vae-layer-conf)))
      (is (= :variational-autoencoder (layer-type {:nn-conf (quick-nn-conf vae-layer-conf)})))
      (is (= org.deeplearning4j.nn.layers.variational.VariationalAutoencoder
             (type (new-layer :nn-conf (quick-nn-conf vae-layer-conf)))))

      ;; forward pass
      (is (= org.deeplearning4j.nn.layers.recurrent.FwdPassReturn (type (new-foward-pass-return))))

      ;; frozen layer
      (is (= org.deeplearning4j.nn.layers.FrozenLayer
             (type (new-frozen-layer
                    (set-param-table!
                     :model (new-layer
                             :nn-conf (quick-nn-conf activation-layer-conf))
                     :param-table-map {"foo" (indarray-of-zeros :rows 1)}))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; nn-conf-builder
;; dl4clj.nn.conf.builders.nn-conf-builder
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest nn-conf-test
  (testing "the creation of neural network configurations"
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
           (type
            (nn-conf-builder
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
             :step-fn (new-gradient-step-fn)
             :convolution-mode :strict
             :lr-policy-power 0.1
             :default-learning-rate-policy :poly
             :build? true))))
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
           (type
            (nn-conf-builder
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
             :build? false))))
    (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
           (type
               (nn-conf-builder :default-activation-fn :relu
                                :step-fn :negative-gradient-step-fn
                                :default-updater :none
                                :use-drop-connect? true
                                :default-drop-out 0.2
                                :default-weight-init :xavier-uniform
                                :build? true
                                :default-gradient-normalization :renormalize-l2-per-layer
                                :layers {0 (dl4clj.nn.conf.builders.builders/dense-layer-builder
                                            :n-in 100
                                            :n-out 1000
                                            :layer-name "first layer"
                                            :activation-fn :tanh
                                            :gradient-normalization :none )
                                         1 {:dense-layer {:n-in 1000
                                                          :n-out 10
                                                          :layer-name "second layer"
                                                          :gradient-normalization :none}}}))))
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration$ListBuilder
           (type
               (nn-conf-builder :default-activation-fn :relu
                                :step-fn :negative-gradient-step-fn
                                :default-updater :none
                                :use-drop-connect? true
                                :default-drop-out 0.2
                                :default-weight-init :xavier-uniform
                                :build? false
                                :default-gradient-normalization :renormalize-l2-per-layer
                                :layers {0 (dl4clj.nn.conf.builders.builders/dense-layer-builder
                                            :n-in 100
                                            :n-out 1000
                                            :layer-name "first layer"
                                            :activation-fn :tanh
                                            :gradient-normalization :none)
                                         1 {:dense-layer {:n-in 1000
                                                          :n-out 10
                                                          :layer-name "second layer"
                                                          :activation-fn :tanh
                                                          :gradient-normalization :none}}}))))
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
           ;; show that the activation fn changes
           (type
               (nn-conf-builder :default-activation-fn :relu
                                :step-fn :negative-gradient-step-fn
                                :default-updater :none
                                :use-drop-connect? true
                                :default-drop-out 0.2
                                :default-weight-init :xavier-uniform
                                :default-gradient-normalization :renormalize-l2-per-layer
                                :build? true
                                :layer (dl4clj.nn.conf.builders.builders/dense-layer-builder
                                          :n-in 100
                                          :n-out 1000
                                          :layer-name "first layer"
                                          :activation-fn :tanh
                                          :gradient-normalization :none)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; list-builder test
;; dl4clj.nn.conf.builders.multi-layer-builders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest list-builder-test
  (testing "the list builder for setting up the :layers key in nn-conf-builder"
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration$ListBuilder
           (type
            (list-builder
             (nn-conf-builder)
             {0 (dl4clj.nn.conf.builders.builders/dense-layer-builder
                 :n-in 100
                 :n-out 1000
                 :layer-name "first layer"
                 :activation-fn :tanh
                 :gradient-normalization :none)
              1 {:dense-layer {:n-in 1000
                               :n-out 10
                               :layer-name "second layer"
                               :activation-fn :tanh
                               :gradient-normalization :none}}}))))
    (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
           (type
            (.build
             (list-builder
              (nn-conf-builder)
              {0 (dl4clj.nn.conf.builders.builders/dense-layer-builder
                  :n-in 100
                  :n-out 1000
                  :layer-name "first layer"
                  :activation-fn :tanh
                  :gradient-normalization :none)
               1 {:dense-layer {:n-in 1000
                                :n-out 10
                                :layer-name "second layer"
                                :activation-fn :tanh
                                :gradient-normalization :none}}})))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; input pre-processors test
;; dl4clj.nn.conf.input-pre-processor
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest pre-processors-test
  (testing "the creation of input preprocessors for use in multi-layer-conf"
    (is (= org.deeplearning4j.nn.conf.preprocessor.BinomialSamplingPreProcessor
           (type (new-binominal-sampling-pre-processor))))
    (is (= org.deeplearning4j.nn.conf.preprocessor.UnitVarianceProcessor
           (type (new-unit-variance-processor))))
    (is (= org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor
           (type (new-rnn-to-cnn-pre-processor :input-height 1 :input-width 1
                                               :num-channels 1))))
    (is (= org.deeplearning4j.nn.conf.preprocessor.ZeroMeanAndUnitVariancePreProcessor
           (type (new-zero-mean-and-unit-variance-pre-processor))))
    (is (= org.deeplearning4j.nn.conf.preprocessor.ZeroMeanPrePreProcessor
           (type (new-zero-mean-pre-pre-processor))))
    (is (= org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor
           (type (new-cnn-to-feed-forward-pre-processor :input-height 1
                                                        :input-width 1
                                                        :num-channels 1))))
    (is (= org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor
           (type (new-cnn-to-rnn-pre-processor :input-height 1 :input-width 1
                                               :num-channels 1))))
    (is (= org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor
           (type (new-feed-forward-to-cnn-pre-processor :input-height 1
                                                        :input-width 1
                                                        :num-channels 1))))
    (is (= org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor
           (type (new-rnn-to-feed-forward-pre-processor))))
    (is (= org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor
           (type (new-feed-forward-to-rnn-pre-processor))))
    (is (= org.deeplearning4j.nn.conf.preprocessor.ComposableInputPreProcessor
           (type (new-composable-input-pre-processor
                  :pre-processors [(new-zero-mean-pre-pre-processor)
                                   (new-binominal-sampling-pre-processor)]))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi-layer-config-builder test
;; dl4clj.nn.conf.builders.multi-layer-builders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest multi-layer-builder-test
  (testing "the creation of mutli layer nn's and setting top level params"
    (let [l-builder (nn-conf-builder :global-activation-fn :relu
                                     :step-fn :negative-gradient-step-fn
                                     :updater :none
                                     :use-drop-connect true
                                     :drop-out 0.2
                                     :weight-init :xavier-uniform
                                     :build? false
                                     :gradient-normalization :renormalize-l2-per-layer
                                     :layers {0 (dl4clj.nn.conf.builders.builders/dense-layer-builder
                                                 :n-in 100
                                                 :n-out 1000
                                                 :layer-name "first layer"
                                                 :activation-fn :tanh
                                                 :gradient-normalization :none)
                                              1 {:dense-layer {:n-in 1000
                                                               :n-out 10
                                                               :layer-name "second layer"
                                                               :activation-fn :tanh
                                                               :gradient-normalization :none}}})
          nn-conf (nn-conf-builder :global-activation-fn :relu
                                   :step-fn :negative-gradient-step-fn
                                   :updater :none
                                   :use-drop-connect true
                                   :drop-out 0.2
                                   :weight-init :xavier-uniform
                                   :gradient-normalization :renormalize-l2-per-layer
                                   :build? true
                                   :layer (dl4clj.nn.conf.builders.builders/dense-layer-builder
                                           :n-in 10
                                           :n-out 100
                                           :layer-name "another layer"
                                           :activation-fn :tanh
                                           :gradient-normalization :none))]
      ;; with a list builder, a built nn-conf, and all opts
      (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
             (type (multi-layer-config-builder
                    :list-builder l-builder
                    :nn-confs nn-conf
                    :backprop? true
                    :backprop-type :standard
                    :input-pre-processors {0 {:zero-mean-pre-pre-processor {}}
                                           1 (new-unit-variance-processor)}
                    :input-type {:feed-forward {:size 100}}
                    :pretrain? false))))
      ;; with no nn-confs and all other opts
      (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
             (type (multi-layer-config-builder
                    :list-builder l-builder
                    :backprop? true
                    :backprop-type :standard
                    :input-pre-processors {0 {:zero-mean-pre-pre-processor {}}
                                           1 (new-unit-variance-processor)}
                    :input-type {:feed-forward {:size 100}}
                    :pretrain? false))))
      ;; with no list-builder but nn-confs and all other opts
      (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
             (type (multi-layer-config-builder
                    :nn-confs [nn-conf nn-conf]
                    :backprop? true
                    :backprop-type :standard
                    :input-pre-processors {0 {:zero-mean-pre-pre-processor {}}
                                           1 (new-unit-variance-processor)}
                    :input-type {:feed-forward {:size 100}}
                    :pretrain? false))))
      ;; with a single nn-conf for nn-confs
      (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
             (type (multi-layer-config-builder
                    :nn-confs nn-conf
                    :backprop? true
                    :backprop-type :standard
                    :input-pre-processors {0 {:zero-mean-pre-pre-processor {}}
                                           1 (new-unit-variance-processor)}
                    :input-type {:feed-forward {:size 100}}
                    :pretrain? false)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi-layer-network test
;; dl4clj.nn.multilayer.multi-layer-network
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest multi-layer-network-creation-test
  (testing "the creation of multi layer networks from multi-layer-confs and their unique methods"
    (let [l-builder (nn-conf-builder :seed 123
                                     :optimization-algo :stochastic-gradient-descent
                                     :iterations 1
                                     :default-learning-rate 0.006
                                     :default-updater :nesterovs
                                     :default-momentum 0.9
                                     :regularization? true
                                     :default-l2 1e-4
                                     :build? false
                                     :default-gradient-normalization :renormalize-l2-per-layer
                                     :layers {0 (dl4clj.nn.conf.builders.builders/dense-layer-builder
                                                 :n-in 784
                                                 :n-out 1000
                                                 :layer-name "first layer"
                                                 :activation-fn :relu
                                                 :weight-init :xavier)
                                              1 {:output-layer {:n-in 1000
                                                                :n-out 10
                                                                :layer-name "second layer"
                                                                :activation-fn :soft-max
                                                                :weight-init :xavier}}})
          mln-conf (multi-layer-config-builder :list-builder l-builder
                                               :backprop? true
                                               :pretrain? false
                                               :build? true)
          mln (new-multi-layer-network :conf mln-conf)
          init-mln (initialize! :mln mln :ds (new-mnist-ds))
          mnist-iter (new-mnist-data-set-iterator :batch-size 128 :train? true :seed 123)
          input (get-features (get-example :ds (new-mnist-ds) :idx 0))
          eval (new-classification-evaler :n-classes 10)
          init-no-ds (init! :model mln)
          evaled (eval-model-whole-ds :mln init-no-ds :iter mnist-iter :evaler eval)]
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
             (type (activate-selected-layers
                    :from 0 :to 1 :mln mln :input input))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (activate-from-prev-layer
                    :current-layer-idx 0 :mln mln :input input :training? true))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (activate-from-prev-layer
                    :current-layer-idx 1 :mln mln :training? true
                    ;; input to second layer is output of first
                    :input (activate-from-prev-layer
                            :current-layer-idx 0 :mln mln :input input :training? true)))))
      (is (= org.deeplearning4j.nn.multilayer.MultiLayerNetwork
             (type (clear-layer-mask-arrays! mln))))
      (is (= java.util.ArrayList
             (type (compute-z :mln init-mln :training? true :input input))))
      (is (= java.util.ArrayList
             (type (compute-z :mln init-mln :training? true))))
      (is (= java.lang.String (type (get-stats :evaler evaled))))
      (is (= org.deeplearning4j.eval.Evaluation
             (type (evaluate-classification :mln init-no-ds :iter mnist-iter))))
      (is (= org.deeplearning4j.eval.Evaluation
             (type
              (evaluate-classification :mln init-no-ds :iter mnist-iter
                                       :labels-list ["0" "1" "2" "3" "4" "5" "6" "7" "8" "9"]))))
      (is (= org.deeplearning4j.eval.RegressionEvaluation
             (type (evaluate-regression :mln init-no-ds :iter mnist-iter))))
      (is (= java.util.ArrayList
             (type (feed-forward :mln init-mln :input input))))
      (is (= java.util.ArrayList
             (type (feed-forward :mln init-mln))))
      (is (= java.util.ArrayList (type (feed-forward-to-layer :mln init-mln :layer-idx 0 :train? true))))
      (is (= java.util.ArrayList (type (feed-forward-to-layer :mln init-mln :layer-idx 0 :input input))))
      (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration (type (get-default-config init-mln))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray (type (get-input init-mln))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray (type (get-labels init-mln))))
      (is (= org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer
             (type (get-layer :mln init-mln :layer-idx 0))))
      (is (= ["first layer" "second layer"] (get-layer-names mln)))
      (is (= (type (array-of :data [] :java-type org.deeplearning4j.nn.api.Layer))
             (type (get-layers init-mln))))
      (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
             (type (get-layer-wise-config init-mln))))
      ;; we never set a mask
      (is (= nil (get-mask init-mln)))
      (is (= 2 (get-n-layers init-mln)))
      (is (= org.deeplearning4j.nn.layers.OutputLayer (type (get-output-layer init-mln))))
      (is (= org.deeplearning4j.nn.updater.MultiLayerUpdater (type (get-updater init-mln))))
      (is (= (type mln) (type (initialize-layers! :mln mln :input input))))
      (is (true? (is-init-called? mln)))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (output :mln init-mln :input input))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (output :mln init-no-ds :iter mnist-iter))))
      (is (= (type mln) (type (print-config mln))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (reconstruct :mln mln
                                :layer-output (first (feed-forward-to-layer
                                                      :layer-idx 0 :mln mln :input input))
                                :layer-idx 1))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (score-examples :mln init-no-ds :iter mnist-iter
                                   :add-regularization-terms? false))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (score-examples :mln init-mln :dataset (new-mnist-ds)
                                   :add-regularization-terms? false))))
      (is (= java.lang.String (type (summary init-mln))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (z-from-prev-layer :mln init-mln :input input
                                   :current-layer-idx 0 :training? true)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fine tuning/transfer learning
;; dl4clj.nn.transfer-learning.fine-tune-conf
;; dl4clj.nn.transfer-learning.helper
;; dl4clj.nn.transfer-learning.transfer-learning
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest transfer-learning-tests
  (testing "the transfer learning fns"
    (let [nn-conf (nn-conf-builder :default-activation-fn :relu
                                   :step-fn :negative-gradient-step-fn
                                   :default-updater :none
                                   :use-drop-connect? false
                                   :default-weight-init :xavier-uniform
                                   :default-gradient-normalization :none
                                   :build? true
                                   :layer (dl4clj.nn.conf.builders.builders/dense-layer-builder
                                           :n-in 10
                                           :n-out 100
                                           :layer-name "some layer"
                                           :activation-fn :tanh
                                           :gradient-normalization :none))
          fine-tune-conf (new-fine-tune-conf :activation-fn :relu
                                             :n-iterations 2
                                             :regularization? false
                                             :seed 1234)
          l-builder (nn-conf-builder :seed 123
                                     :optimization-algo :stochastic-gradient-descent
                                     :iterations 1
                                     :default-learning-rate 0.006
                                     :default-updater :nesterovs
                                     :default-momentum 0.9
                                     :regularization? false
                                     :build? false
                                     :default-gradient-normalization :none
                                     :layers {0 (dl4clj.nn.conf.builders.builders/dense-layer-builder
                                                 :n-in 784
                                                 :n-out 1000
                                                 :layer-name "first layer"
                                                 :activation-fn :relu
                                                 :weight-init :xavier)
                                              1 {:output-layer {:n-in 1000
                                                                :n-out 10
                                                                :layer-name "second layer"
                                                                :activation-fn :soft-max
                                                                :weight-init :xavier}}})
          mln-conf (multi-layer-config-builder :list-builder l-builder
                                               :backprop? true
                                               :pretrain? false
                                               :build? true)
          mln (init! :model (new-multi-layer-network :conf mln-conf))
          helper (new-helper :mln mln :frozen-til 0)
          featurized (featurize :helper helper :data-set (get-example :ds (new-mnist-ds)
                                                                      :idx 0))
          featurized-input (get-features featurized)
          tlb (transfer-learning-builder
               :mln (init!
                     :model
                     (new-multi-layer-network
                      :conf
                      (multi-layer-config-builder
                       :list-builder (nn-conf-builder
                                      :seed 123
                                      :optimization-algo :stochastic-gradient-descent
                                      :iterations 1
                                      :default-learning-rate 0.006
                                      :default-updater :nesterovs
                                      :default-momentum 0.9
                                      :regularization? false
                                      :build? false
                                      :default-gradient-normalization :none
                                      :layers {0 (dense-layer-builder
                                                  :n-in 10
                                                  :n-out 100
                                                  :layer-name "first layer"
                                                  :activation-fn :relu
                                                  :weight-init :xavier)
                                               1 {:activation-layer {:n-in 100
                                                                     :n-out 10
                                                                     :layer-name "second layer"
                                                                     :activation-fn :soft-max
                                                                     :weight-init :xavier}}
                                               2 {:output-layer {:n-in 10
                                                                 :n-out 1
                                                                 :layer-name "output layer"
                                                                 :activation-fn :soft-max
                                                                 :weight-init :xavier}}})
                       :backprop? true
                       :pretrain? false
                       :build? true)))
               :build? false)]
      ;; dl4clj.nn.transfer-learning.fine-tune-conf
      (is (= org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
             (type fine-tune-conf)))
      (is (= org.deeplearning4j.nn.transferlearning.FineTuneConfiguration$Builder
             (type (new-fine-tune-conf :activation-fn :relu
                                       :n-iterations 2
                                       :regularization? true
                                       :seed 123
                                       :build? false))))
      (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
             (type (applied-to-nn-conf! :fine-tune-conf fine-tune-conf
                                        :nn-conf nn-conf))))
      (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
             (type (nn-conf-from-fine-tune-conf :fine-tune-conf fine-tune-conf))))
      (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
             (type (nn-conf-from-fine-tune-conf :fine-tune-conf fine-tune-conf
                                                :build? true))))
      ;; dl4clj.nn.transfer-learning.helper
      (is (= org.deeplearning4j.nn.transferlearning.TransferLearningHelper
             (type (new-helper :mln mln))))
      (is (= org.deeplearning4j.nn.transferlearning.TransferLearningHelper
             (type helper)))
      (is (= org.nd4j.linalg.dataset.DataSet (type featurized)))
      (is (= org.deeplearning4j.nn.transferlearning.TransferLearningHelper
             (type (fit-featurized! :helper helper :data-set featurized))))
      (is (= org.nd4j.linalg.cpu.nativecpu.NDArray
             (type (output-from-featurized :helper helper :featurized-input featurized-input))))
      (is (= org.deeplearning4j.nn.multilayer.MultiLayerNetwork
             (type (unfrozen-mln helper))))

      ;; dl4clj.nn.transfer-learning.transfer-learning
      (is (= org.deeplearning4j.nn.multilayer.MultiLayerNetwork
             (type (transfer-learning-builder
                    :mln mln
                    :fine-tune-conf fine-tune-conf
                    :remove-output-layer? true
                    :replacement-layer {:layer-idx 0 :n-out 1001 :weight-init :xavier-uniform}
                    :remove-last-n-layers 1
                    :add-layer (dl4clj.nn.conf.builders.builders/output-layer-builder
                            :n-in 100
                            :n-out 10
                            :layer-name "another layer"
                            :activation-fn :tanh
                            :gradient-normalization :none)
                    :set-feature-extractor-idx 0
                    :input-pre-processor {:layer-idx 0 :pre-processor (new-unit-variance-processor)}))))
      ;; testing add-layers
      (is (= ["first layer" "replacement another layer" "replacement second layer"]
             (get-layer-names
              (transfer-learning-builder :tlb tlb
                                         :fine-tune-conf fine-tune-conf
                                         :remove-last-n-layers 2
                                         :add-layers {1 (dl4clj.nn.conf.builders.builders/activation-layer-builder
                                                         :n-in 10
                                                         :n-out 100
                                                         :layer-name "replacement another layer"
                                                         :activation-fn :tanh
                                                         :gradient-normalization :none)
                                                      2 {:output-layer {:n-in 100
                                                                        :n-out 10
                                                                        :layer-name "replacement second layer"
                                                                        :activation-fn :soft-max
                                                                        :weight-init :xavier}}}))))
      ;; testing add-layer
      (is (= ["first layer" "another layer"]
             (get-layer-names
              (transfer-learning-builder
               :mln mln
               :fine-tune-conf fine-tune-conf
               :remove-output-layer? true
               :add-layer (dl4clj.nn.conf.builders.builders/output-layer-builder
                           :n-in 100
                           :n-out 10
                           :layer-name "another layer"
                           :activation-fn :tanh
                           :gradient-normalization :none)
               :set-feature-extractor-idx 0
               :input-pre-processor {:layer-idx 0 :pre-processor (new-unit-variance-processor)})))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; dl4clj.nn.updater.layer-updater
;; dl4clj.nn.updater.multi-layer-updater
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest updater-tests
  (testing "the creation of model updaters"
    (let [l-builder (nn-conf-builder
                     :seed 123
                     :optimization-algo :stochastic-gradient-descent
                     :iterations 1
                     :learning-rate 0.006
                     :updater :nesterovs
                     :momentum 0.9
                     :regularization? false
                     :build? false
                     :gradient-normalization :none
                     :layers {0 (dense-layer-builder
                                 :n-in 10
                                 :n-out 100
                                 :layer-name "first layer"
                                 :activation-fn :relu
                                 :weight-init :xavier)
                              1 {:activation-layer {:n-in 100
                                                    :n-out 10
                                                    :layer-name "second layer"
                                                    :activation-fn :soft-max
                                                    :weight-init :xavier}}
                              2 {:output-layer {:n-in 10
                                                :n-out 1
                                                :layer-name "output layer"
                                                :activation-fn :soft-max
                                                :weight-init :xavier}}})
          mln-conf (multi-layer-config-builder
                    :list-builder l-builder
                    :backprop? true
                    :pretrain? false
                    :build? true)
          mln (init! :model (new-multi-layer-network :conf mln-conf))
          layer-updater (new-layer-updater)
          layer (as-> (graves-lstm-layer-builder
                       :n-in 10 :n-out 2 :forget-gate-bias-init 0.2
                       :gate-activation-fn :relu
                       :n-out 2
                       :activation-fn :relu
                       :bias-init 0.7 :bias-learning-rate 0.1
                       :dist {:normal {:mean 0 :std 1}}
                       :drop-out 0.2 :epsilon 0.3
                       :gradient-normalization :renormalize-l2-per-layer
                       :l2 0.1 :l2-bias 1
                       :gradient-normalization-threshold 0.9
                       :layer-name "foo"
                       :learning-rate 0.1 :learning-rate-policy :inverse
                       :learning-rate-schedule {0 0.6 1 0.5}
                       :momentum 0.2  :momentum-after {0 0.3 1 0.4}
                       :updater :nesterovs :weight-init :distribution) l
                  (nn-conf-builder :layer l :regularization? true :build? true)
                  ;; I can add variables now
                  ;; just can't add anything to l1/2ByParam
                  ;; going to need to add this logic to add-variable
                  (do (.variables l (.add (.variables l false) "baz"))
                      l)
                  (add-variable! :nn-conf l :var-name "baz")
                  (set-learning-rate-by-param! :nn-conf l :var-name "foo" :rate 0.2)
                  (new-layer :nn-conf l)
                  ;; try initializing the layer instead of calling new layer for setting l2byparam
                  ;; was able to set that in a mln above
                  ;; multi-layer-network-creation-test
                  )]
      (is (= org.deeplearning4j.nn.updater.LayerUpdater (type layer-updater)))
      (is (= org.deeplearning4j.nn.updater.MultiLayerUpdater
             (type (new-multi-layer-updater :mln mln))))
      (is (= (type layer)
             (type (:layer (apply-lrate-decay-policy! :updater layer-updater
                                              :layer layer
                                              :iteration 1
                                              :variable "foo"
                                              :decay-policy :score)))))
      (is (= (type layer)
             (type (:layer (apply-momentum-decay-policy! :updater layer-updater
                                                         :layer layer :iteration 1
                                                         :variable "foo")))))
      (is (= (type layer)
             (type (:layer (pre-apply! :updater layer-updater
                                       :layer layer :iteration 1
                                       :gradient (new-default-gradient))))))
      (is (= {} (get-updater-for-variable layer-updater)))
      ;; cant get this to work, cant add things to the l2ByParam hash map for some damn reason
      ;; thats what this is trying to do under the hood
      ;; https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/conf/NeuralNetConfiguration.java
      #_(is (= "" (post-apply! :updater layer-updater
                             :layer layer :gradient-array (rand [2])
                             :param "foo" :mini-batch-size 10))))))



(comment

  (dl4clj.nn.conf.layers.shared-fns/instantiate
 :layer (dl4clj.nn.conf.builders.builders/activation-layer-builder
         :n-in 10
         :n-out 100
         :layer-name "another layer"
         :activation-fn :tanh
         :gradient-normalization :none)
 :conf
 (nn-conf-builder :global-activation-fn :relu
                  :step-fn :negative-gradient-step-fn
                  :updater :none
                  :use-drop-connect true
                  :drop-out 0.2
                  :weight-init :xavier-uniform
                  :gradient-normalization :renormalize-l2-per-layer
                  :build? true
                  :layer (dl4clj.nn.conf.builders.builders/activation-layer-builder
                          :n-in 10
                          :n-out 100
                          :layer-name "another layer"
                          :activation-fn :tanh
                          :gradient-normalization :none))
 :listeners (dl4clj.optimize.listeners.listeners/new-score-iteration-listener)
 #_(dl4clj.utils/array-of :data (dl4clj.optimize.listeners.listeners/new-score-iteration-listener)
                                   :java-type org.deeplearning4j.optimize.api.IterationListener)
 :layer-idx 0
 :layer-param-view (nd4clj.linalg.factory.nd4j/rand [10])
 :initialize-params? true))
