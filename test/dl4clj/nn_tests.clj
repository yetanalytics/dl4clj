(ns dl4clj.nn-tests
  (:require [dl4clj.nn.conf.builders.builders :refer :all]
            [clojure.test :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; any layer builder
;; dl4clj.nn.conf.builders.builders
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest layer-builder-test
  (testing "the creation of any layer in dl4j"
    (is (= org.deeplearning4j.nn.conf.layers.ActivationLayer
           (type
            (activation-layer-builder :n-in 10 :n-out 2 :activation-fn :relu
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



    ))
