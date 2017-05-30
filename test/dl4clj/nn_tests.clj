(ns dl4clj.nn-tests
  (:require [dl4clj.nn.conf.builders.builders :refer :all]
            [clojure.test :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; any layer builder
;; dl4clj.nn.conf.builders.builders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest layer-builder-test
  (testing "the creation of any layer in dl4j"
    ;; activation layer
    {:activation-fn :relu
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
     :weight-init :distribution}
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
             :alpha 0.1 :gradient-check? true :lambda 0.1
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
             :hidden-unit :softmax :visible-unit :gaussian
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
             :is-mini-batch true :lock-gamma-beta? true
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
    (is (= ""
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
             :weight-init :distribution
             )
            )))






    ))
