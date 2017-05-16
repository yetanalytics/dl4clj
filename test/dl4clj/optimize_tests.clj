(ns dl4clj.optimize-tests
  (:require [dl4clj.optimize.solver :refer :all]
            [dl4clj.optimize.solvers.optimizers :refer :all]
            [dl4clj.optimize.step-functions.step-fns :refer :all]
            [dl4clj.nn.conf.builders.nn-conf-builder :as nn]
            [dl4clj.nn.conf.builders.multi-layer-builders :as mlb]
            [dl4clj.optimize.listeners.listeners :refer :all]
            [clojure.test :refer :all]
            [dl4clj.nn.layers.recurrent.graves-bidirectional-lstm :refer [new-bidirectional-lstm-layer]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; objects that I need for testing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def nn-conf
  (-> (nn/nn-conf-builder
       {:seed 123
        :optimization-algo :stochastic-gradient-descent
        :iterations 1
        :learning-rate 0.006
        :updater :nesterovs
        :momentum 0.9
        :regularization true
        :l2 1e-4
        :layer {:graves-lstm
                {:n-in 10
                 :n-out 1000
                 :updater :rmsprop
                 :activation :tanh
                 :weight-init :distribution
                 :dist {:uniform {:lower -0.08, :upper 0.08}}}}
        :pretrain false
        :backprop true})
   (mlb/multi-layer-config-builder {})))

(def single-listener (new-score-iteration-listener :print-every-n 2))

(def multiple-listeners [(new-score-iteration-listener :print-every-n 2)
                         (new-collection-scores-iteration-listener :frequency 2)])

(def model (new-bidirectional-lstm-layer :conf nn-conf))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of generic solvers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest generic-solvers
  ;; still need to write a good test for optimize!
  (testing "the creation of solvers and unit tests for their methods"
    (is (= org.deeplearning4j.optimize.Solver (type (build-solver
                                                     :nn-conf nn-conf))))
    (is (= org.deeplearning4j.optimize.Solver (type (build-solver
                                                     :nn-conf nn-conf
                                                     :single-listener single-listener))))
    (is (= org.deeplearning4j.optimize.Solver (type (build-solver
                                                     :nn-conf nn-conf
                                                     :model model))))
    (is (= org.deeplearning4j.optimize.Solver (type (build-solver
                                                     :nn-conf nn-conf
                                                     :multiple-listeners multiple-listeners))))
    (is (= org.deeplearning4j.optimize.Solver (type (build-solver
                                                     :nn-conf nn-conf
                                                     :model model
                                                     :single-listener single-listener))))
    (is (= org.deeplearning4j.optimize.Solver (type (build-solver
                                                     :nn-conf nn-conf
                                                     :model model
                                                     :multiple-listeners multiple-listeners))))
    (is (= org.deeplearning4j.optimize.Solver (type (build-solver
                                                     :single-listener single-listener
                                                     :multiple-listeners multiple-listeners
                                                     :model model
                                                     :nn-conf nn-conf))))
    (is (= org.deeplearning4j.optimize.solvers.StochasticGradientDescent
           (type (get-optimizer (build-solver
                                 :nn-conf nn-conf
                                 :model model
                                 :single-listener single-listener)))))
    (is (= org.deeplearning4j.optimize.Solver
           ;; need to find a good way to test that the new listeners are added
           ;; not sure if that will be possible tho
           (type (set-listeners! :solver (build-solver
                                          :nn-conf nn-conf
                                          :model model
                                          :single-listener single-listener)
                                 :listeners multiple-listeners))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of step fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest step-functions
  (testing "the creation of step functions and that the fns implement the step function interface"
    (is (= org.deeplearning4j.optimize.stepfunctions.DefaultStepFunction
           (type (step-fns :default))))
    (is (= org.deeplearning4j.optimize.stepfunctions.DefaultStepFunction
           (type (new-step-fn-from-nn-conf-step-fn :nn-conf-step-fn :default-step-fn))))

    (is (= org.deeplearning4j.optimize.stepfunctions.GradientStepFunction
           (type (step-fns :gradient))))
    (is (= org.deeplearning4j.optimize.stepfunctions.GradientStepFunction
           (type (new-step-fn-from-nn-conf-step-fn :nn-conf-step-fn :gradient-step-fn))))

    (is (= org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction
           (type (step-fns :negative-default))))
    (is (= org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction
           (type (new-step-fn-from-nn-conf-step-fn :nn-conf-step-fn :negative-default-step-fn))))


    (is (= org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction
           (type (step-fns :negative-gradient))))
    (is (= org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction
           (type (new-step-fn-from-nn-conf-step-fn :nn-conf-step-fn :negative-gradient-step-fn))))

    ))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of optimizers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest various-optimizers
  (testing "the creation of optimizers and that the created optimizers implement the correct interfaces"
    (is (= "" (optimizers {:base {:nn-conf nn-conf}})))



    ))
