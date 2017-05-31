(ns dl4clj.nn-tests
  (:require [dl4clj.nn.conf.builders.builders :refer :all]
            [dl4clj.nn.conf.builders.nn-conf-builder :refer :all]
            [dl4clj.nn.conf.builders.multi-layer-builders :refer :all]
            [dl4clj.nn.conf.input-pre-processor :refer :all]
            [dl4clj.nn.conf.constants :refer :all]
            [dl4clj.nn.conf.distribution.distribution :refer :all]
            [clojure.test :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; constants, value-of, input-type
;; dl4clj.nn.conf.constants
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest enum-testing
  (testing "the creation of dl4j enums/constants"
    ;; constants (helper fn)
    (is (= "RELU" (constants #(str %) :relu :activation? true)))
    (is (= "FooBaz" (constants #(str %) :foo-baz :camel? true)))
    (is (= "Relu" (constants #(str %) :relu :activation? true :camel? true)))
    (is (= "FOO_BAZ" (constants #(str %) :foo-baz)))

    (is (= org.nd4j.linalg.activations.Activation
           (type (value-of {:activation-fn :relu}))))
    (is (= org.deeplearning4j.nn.conf.GradientNormalization
           (type (value-of {:gradient-normalization :none}))))
    (is (= org.deeplearning4j.nn.conf.LearningRatePolicy
           (type (value-of {:learning-rate-policy :poly}))))
    (is (= org.deeplearning4j.nn.conf.Updater
           (type (value-of {:updater :adam}))))
    (is (= org.deeplearning4j.nn.weights.WeightInit
           (type (value-of {:weight-init :xavier}))))
    (is (= org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction
           (type (value-of {:loss-fn :mse}))))
    (is (= org.deeplearning4j.nn.conf.layers.RBM$HiddenUnit
           (type (value-of {:hidden-unit :binary}))))
    (is (= org.deeplearning4j.nn.conf.layers.RBM$VisibleUnit
           (type (value-of {:visible-unit :binary}))))
    (is (= org.deeplearning4j.nn.conf.ConvolutionMode
           (type (value-of {:convolution-mode :strict}))))
    (is (= org.deeplearning4j.nn.conf.layers.ConvolutionLayer$AlgoMode
           (type (value-of {:cudnn-algo-mode :no-workspace}))))
    (is (= org.deeplearning4j.nn.conf.layers.PoolingType
           (type (value-of {:pool-type :avg}))))
    (is (= org.deeplearning4j.nn.conf.BackpropType
           (type (value-of {:backprop-type :standard}))))
    (is (= org.deeplearning4j.nn.api.OptimizationAlgorithm
           (type (value-of {:optimization-algorithm :lbfgs}))))
    (is (= org.deeplearning4j.nn.api.MaskState
           (type (value-of {:mask-state :active}))))
    (is (= org.deeplearning4j.nn.api.Layer$Type
           (type (value-of {:layer-type :feed-forward}))))
    (is (= org.deeplearning4j.nn.api.Layer$TrainingMode
           (type (value-of {:layer-training-mode :train}))))

    (is (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeRecurrent
           (type (input-types {:recurrent {:size 10}}))))
    (is (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeFeedForward
           (type (input-types {:feed-forward {:size 10}}))))
    (is (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeConvolutional
           (type (input-types {:convolutional {:height 1 :width 1 :depth 1}}))))
    (is (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeConvolutionalFlat
           (type (input-types {:convolutional-flat {:height 1 :width 1 :depth 1}}))))))

(deftest distributions-test
  (testing "the creation of distributions for use in a nn-conf"
    (is (= org.deeplearning4j.nn.conf.distribution.UniformDistribution
           (type (new-uniform-distribution :lower 0.2 :upper 0.4))))
    (is (= org.deeplearning4j.nn.conf.distribution.NormalDistribution
           (type (new-normal-distribution :mean 0 :std 1))))
    (is (= org.deeplearning4j.nn.conf.distribution.GaussianDistribution
           (type (new-gaussian-distribution :mean 0.0 :std 1))))
    (is (= org.deeplearning4j.nn.conf.distribution.BinomialDistribution
           (type (new-binomial-distribution :number-of-trials 2
                                            :probability-of-success 0.5))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; any layer builder
;; dl4clj.nn.conf.builders.builders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest layer-builder-test
  (testing "the creation of any layer in dl4j"
    (is (= org.deeplearning4j.nn.conf.layers.ActivationLayer
           (type
            (activation-layer-builder
             :n-in 10 :n-out 2 :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer
           (type
            (center-loss-output-layer-builder
             :alpha 0.1 :gradient-check? false :lambda 0.1
             :loss-fn :mse
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.OutputLayer
           (type
            (output-layer-builder
             :n-in 10 :n-out 2 :loss-fn :mse
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.RnnOutputLayer
           (type
            (rnn-output-layer-builder
             :n-in 10 :n-out 2 :loss-fn :mse
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.AutoEncoder
           (type
            (auto-encoder-layer-builder
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
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.RBM
           (type
            (rbm-layer-builder
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
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM
           (type
            (graves-bidirectional-lstm-layer-builder
             :n-in 10 :n-out 2 :forget-gate-bias-init 0.2
             :gate-activation-fn :relu
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.GravesLSTM
           (type
            (graves-lstm-layer-builder
             :n-in 10 :n-out 2 :forget-gate-bias-init 0.2
             :gate-activation-fn :relu
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.BatchNormalization
           (type
            (batch-normalization-layer-builder
             :n-in 10 :n-out 2 :beta 0.5
             :decay 0.3 :eps 0.1 :gamma 0.1
             :mini-batch? false :lock-gamma-beta? true
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.ConvolutionLayer
           (type
            (convolutional-layer-builder
             :n-in 10 :n-out 2
             :kernel-size [2 2] :padding [2 2] :stride [2 2]
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.Convolution1DLayer
           (type
            (convolution-1d-layer-builder
             :n-in 10 :n-out 2
             :kernel-size 6 :stride 3 :padding 3
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.DenseLayer
           (type
            (dense-layer-builder
             :n-in 10 :n-out 2
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.EmbeddingLayer
           (type
            (embedding-layer-builder
             :n-in 10 :n-out 2
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.LocalResponseNormalization
           (type
            (local-response-normalization-layer-builder
             :alpha 0.2 :beta 0.2 :k 0.2 :n 1
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.SubsamplingLayer
           (type
            (subsampling-layer-builder
             :kernel-size [2 2] :stride [2 2] :padding [2 2]
             :pooling-type :sum
             :build? true
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.Subsampling1DLayer
           (type
            (subsampling-1d-layer-builder
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
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.LossLayer
           (type
            (loss-layer-builder
             :loss-fn :mse
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.DropoutLayer
           (type
            (dropout-layer-builder
             :n-in 2 :n-out 10
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer
           (type
            (global-pooling-layer-builder
             :pooling-dimensions [3 2]
             :collapse-dimensions? true
             :pnorm 2
             :pooling-type :pnorm
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.ZeroPaddingLayer
           (type
            (zero-padding-layer-builder
             :pad-top 1 :pad-bot 2 :pad-left 3 :pad-right 4
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))
    (is (= org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder
           (type
            (variational-autoencoder-builder
             :n-in 5 :n-out 10 :loss-fn :mse
             :pre-train-iterations 1 :visible-bias-init 2
             :decoder-layer-sizes [5 9]
             :encoder-layer-sizes [7 2]
             :reconstruction-distribution {:gaussian {:activation-fn :tanh}}
             :vae-loss-fn {:output-activation-fn :tanh :loss-fn :mse}
             :num-samples 2 :pzx-activation-function :tanh
             :activation-fn :relu
             :adam-mean-decay 0.2 :adam-var-decay 0.1
             :bias-init 0.7 :bias-learning-rate 0.1
             :dist {:normal {:mean 0 :std 1}}
             :drop-out 0.2 :epsilon 0.3
             :gradient-normalization :none
             :gradient-normalization-threshold 0.9
             :l1 0.2 :l2 0.7 :layer-name "foo"
             :learning-rate 0.1 :learning-rate-policy :inverse
             :l1-bias 0.1 :l2-bias 0.2
             :learning-rate-schedule {0 0.2 1 0.5}
             :momentum 0.2 :momentum-after {0 0.3 1 0.4}
             :rho 0.7 :rms-decay 0.7 :updater :adam
             :weight-init :distribution))))))

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
             :lr-policy-power 0.4
             :learning-rate-policy :poly
             :max-num-line-search-iterations 6
             :mini-batch? true
             :minimize? true
             :use-drop-connect? true
             :optimization-algo :lbfgs
             :lr-score-based-decay-rate 0.7
             :regularization? true
             :seed 123
             :step-fn :default-step-fn
             :convolution-mode :strict
             :build? true))))
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder
           (type
            (nn-conf-builder
             :iterations 1
             :lr-policy-decay-rate 0.3
             :lr-policy-power 0.4
             :learning-rate-policy :poly
             :max-num-line-search-iterations 6
             :mini-batch? true
             :minimize? true
             :use-drop-connect? true
             :optimization-algo :lbfgs
             :lr-score-based-decay-rate 0.7
             :regularization? true
             :seed 123
             :step-fn :default-step-fn
             :convolution-mode :strict
             :build? false))))
    (is (= org.deeplearning4j.nn.conf.MultiLayerConfiguration
           (type
               (nn-conf-builder :global-activation-fn :relu
                                :step-fn :negative-gradient-step-fn
                                :updater :none
                                :use-drop-connect true
                                :drop-out 0.2
                                :weight-init :xavier-uniform
                                :build? true
                                :gradient-normalization :renormalize-l2-per-layer
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
               (nn-conf-builder :global-activation-fn :relu
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
                                            :gradient-normalization :none )
                                         1 {:dense-layer {:n-in 1000
                                                          :n-out 10
                                                          :layer-name "second layer"
                                                          :activation-fn :tanh
                                                          :gradient-normalization :none}}}))))
    (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
           (type
               (nn-conf-builder :global-activation-fn :relu
                                :step-fn :negative-gradient-step-fn
                                :updater :none
                                :use-drop-connect true
                                :drop-out 0.2
                                :weight-init :xavier-uniform
                                :gradient-normalization :renormalize-l2-per-layer
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
