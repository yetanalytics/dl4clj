(ns dl4clj.optimize-tests
  (:refer-clojure :exclude [rand max])
  (:require [dl4clj.optimize.solver :refer :all]
            [dl4clj.optimize.solvers.optimizers :refer :all]
            [dl4clj.optimize.step-functions.step-fns :refer :all]
            [dl4clj.nn.conf.builders.nn-conf-builder :as nn]
            [dl4clj.nn.conf.builders.multi-layer-builders :as mlb]
            [dl4clj.nn.conf.builders.builders :as l]
            [dl4clj.optimize.listeners.listeners :refer :all]
            [clojure.test :refer :all]
            [dl4clj.optimize.api.convex-optimizer :refer :all]
            [dl4clj.utils :refer [array-of]]
            [dl4clj.optimize.termination.terminations :refer :all]
            [dl4clj.nn.layers.layer-creation :refer [new-layer]]
            [nd4clj.linalg.factory.nd4j :refer :all]
            [dl4clj.nn.updater.layer-updater :refer [new-layer-updater]]
            [dl4clj.nn.gradient.default-gradient :refer [new-default-gradient]]
            [dl4clj.optimize.api.line-optimizer :refer :all]
            [dl4clj.optimize.api.step-fn :refer :all]
            [dl4clj.optimize.api.termination-condition :refer :all]
            [dl4clj.optimize.api.iteration-listener :refer :all]
            [clojure.java.io :as io])
  (:import [org.deeplearning4j.optimize.api IterationListener]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; objects that I need for testing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def nn-conf
  (nn/nn-conf-builder
   :seed 123
   :optimization-algo :stochastic-gradient-descent
   :iterations 1
   :learning-rate 0.006
   :updater :nesterovs
   :momentum 0.9
   :build? true
   :regularization true
   :l2 1e-4
   :layer {:graves-lstm
           {:n-in 10
            :n-out 1000
            :updater :rmsprop
            :activation :tanh
            :weight-init :distribution
            :dist {:uniform {:lower -0.08, :upper 0.08}}}}))

(def nn-conf-unsupervised
  (nn/nn-conf-builder
   :seed 123
   :iterations 1
   :optimization-algo :stochastic-gradient-descent
   :learning-rate 1e-2
   :updater :rmsprop
   :rms-decay 0.95
   :weight-init :xavier
   :regularization true
   :build? true
   :l2 1e-4
   :layer {:variational-auto-encoder {:activation-fn :leaky-relu
                                      :encoder-layer-sizes [256 256]
                                      :decoder-layer-sizes [256 256]
                                      :pzx-activation-function :identity
                                      :reconstruction-distribution {:bernoulli {:activation-fn :sigmoid}}
                                      :n-in 5
                                      :n-out 2}}))

(def single-listener (new-score-iteration-listener :print-every-n 2 :array? true))

(def multiple-listeners [(new-score-iteration-listener :print-every-n 2)
                         (new-collection-scores-iteration-listener :frequency 2)])

(def model (new-layer :nn-conf nn-conf))

(def unsupervised-model (new-layer :nn-conf nn-conf-unsupervised))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of generic solvers
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/package-summary.html
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
                                                     :nn-conf nn-conf))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of step fns
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/stepfunctions/package-summary.html
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
           (type (new-step-fn-from-nn-conf-step-fn :nn-conf-step-fn :negative-gradient-step-fn))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of listeners
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/listeners/package-summary.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest listeners-test
  (testing "the creation of iteration listeners"
    (is (= org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener
           (type (listeners {:param-and-gradient {}}))))
    (is (= org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener
           (type (new-param-and-gradient-iteration-listener))))

    (is (= org.deeplearning4j.optimize.listeners.CollectScoresIterationListener
           (type (listeners {:collection-scores {}}))))
    (is (= org.deeplearning4j.optimize.listeners.CollectScoresIterationListener
           (type (listeners {:collection-scores {:frequency 5}}))))
    (is (= org.deeplearning4j.optimize.listeners.CollectScoresIterationListener
           (type (new-collection-scores-iteration-listener :frequency 5))))
    (is (= org.deeplearning4j.optimize.listeners.CollectScoresIterationListener
           (type (new-collection-scores-iteration-listener))))

    (is (= org.deeplearning4j.optimize.listeners.ComposableIterationListener
           (type (listeners {:composable {:listeners multiple-listeners}}))))
    (is (= org.deeplearning4j.optimize.listeners.ComposableIterationListener
           (type (new-composable-iteration-listener :coll-of-listeners multiple-listeners))))

    (is (= org.deeplearning4j.optimize.listeners.ScoreIterationListener
           (type (listeners {:score-iteration {}}))))
    (is (= org.deeplearning4j.optimize.listeners.ScoreIterationListener
           (type (listeners {:score-iteration {:print-every-n 5}}))))
    (is (= org.deeplearning4j.optimize.listeners.ScoreIterationListener
           (type (new-score-iteration-listener :print-every-n 5))))
    (is (= org.deeplearning4j.optimize.listeners.ScoreIterationListener
           (type (new-score-iteration-listener))))

    (is (= org.deeplearning4j.optimize.listeners.PerformanceListener
           (type (listeners {:performance {:build? true}}))))
    (is (= org.deeplearning4j.optimize.listeners.PerformanceListener
           (type (new-performance-iteration-listener))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of termination conditions
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/terminations/package-summary.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest terminations
  (testing "the creation of termination conditions"
    (is (= org.deeplearning4j.optimize.terminations.EpsTermination
           (type (termination-condition {:eps {}}))))
    (is (= org.deeplearning4j.optimize.terminations.EpsTermination
           (type (termination-condition {:eps {:eps 1.0 :tolerance 2.0}}))))
    (is (= org.deeplearning4j.optimize.terminations.EpsTermination
           (type (new-eps-termination-condition :eps 1.0 :tolerance 2.0))))
    (is (= org.deeplearning4j.optimize.terminations.EpsTermination
           (type (new-eps-termination-condition))))

    (is (= org.deeplearning4j.optimize.terminations.Norm2Termination
           (type (termination-condition {:norm-2 {:gradient-tolerance 2.0}}))))
    (is (= org.deeplearning4j.optimize.terminations.Norm2Termination
           (type (new-norm-2-termination-condition :gradient-tolerance 2.0))))

    (is (= org.deeplearning4j.optimize.terminations.ZeroDirection
           (type (termination-condition {:zero-direction {}}))))
    (is (= org.deeplearning4j.optimize.terminations.ZeroDirection
           (type (new-zero-direction-termination-condition))))

    (is (= org.deeplearning4j.optimize.terminations.TerminationConditions
           (type (termination-condition {:default {}}))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of optimizers
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/solvers/package-summary.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest various-optimizers
  (testing "the creation of optimizers"
    (is (= org.deeplearning4j.optimize.solvers.ConjugateGradient
           (type (optimizers {:conjugate-geradient
                              {:nn-conf nn-conf
                               :step-fn (step-fns :gradient)
                               :listeners multiple-listeners
                               :termination-condition [(new-zero-direction-termination-condition)]
                               :model model}}))))
    (is (= org.deeplearning4j.optimize.solvers.ConjugateGradient
           (type (new-conjugate-gradient-optimizer :nn-conf nn-conf
                                                   :step-fn (step-fns :gradient)
                                                   :listeners multiple-listeners
                                                   :termination-condition [(new-zero-direction-termination-condition)]
                                                   :model model))))
    (is (= org.deeplearning4j.optimize.solvers.ConjugateGradient
           (type (optimizers {:conjugate-geradient
                              {:nn-conf nn-conf
                               :step-fn (step-fns :gradient)
                               :listeners multiple-listeners
                               :model model}}))))
    (is (= org.deeplearning4j.optimize.solvers.ConjugateGradient
           (type (new-conjugate-gradient-optimizer :nn-conf nn-conf
                                                   :step-fn (step-fns :gradient)
                                                   :listeners multiple-listeners
                                                   :model model))))

    (is (= org.deeplearning4j.optimize.solvers.LBFGS
           (type (optimizers {:lbfgs
                              {:nn-conf nn-conf
                               :step-fn (step-fns :gradient)
                               :listeners multiple-listeners
                               :termination-condition [(new-zero-direction-termination-condition)]
                               :model model}}))))
    (is (= org.deeplearning4j.optimize.solvers.LBFGS
           (type (new-lbfgs-optimizer :nn-conf nn-conf
                                      :step-fn (step-fns :gradient)
                                      :listeners multiple-listeners
                                      :termination-condition [(new-zero-direction-termination-condition)]
                                      :model model))))
    (is (= org.deeplearning4j.optimize.solvers.LBFGS
         (type (optimizers {:lbfgs
                              {:nn-conf nn-conf
                               :step-fn (step-fns :gradient)
                               :listeners multiple-listeners
                               :model model}}))))
    (is (= org.deeplearning4j.optimize.solvers.LBFGS
           (type (new-lbfgs-optimizer :nn-conf nn-conf
                                      :step-fn (step-fns :gradient)
                                      :listeners multiple-listeners
                                      :model model))))

    (is (= org.deeplearning4j.optimize.solvers.LineGradientDescent
           (type (optimizers {:line-gradient-descent
                              {:nn-conf nn-conf
                               :step-fn (step-fns :gradient)
                               :listeners multiple-listeners
                               :termination-condition [(new-zero-direction-termination-condition)]
                               :model model}}))))
    (is (= org.deeplearning4j.optimize.solvers.LineGradientDescent
           (type (new-line-gradient-descent-optimizer :nn-conf nn-conf
                                                      :step-fn (step-fns :gradient)
                                                      :listeners multiple-listeners
                                                      :termination-condition [(new-zero-direction-termination-condition)]
                                                      :model model))))
    (is (= org.deeplearning4j.optimize.solvers.LineGradientDescent
           (type (optimizers {:line-gradient-descent
                              {:nn-conf nn-conf
                               :step-fn (step-fns :gradient)
                               :listeners multiple-listeners
                               :model model}}))))
    (is (= org.deeplearning4j.optimize.solvers.LineGradientDescent
           (type (new-line-gradient-descent-optimizer :nn-conf nn-conf
                                                      :step-fn (step-fns :gradient)
                                                      :listeners multiple-listeners
                                                      :model model))))

    (is (= org.deeplearning4j.optimize.solvers.StochasticGradientDescent
           (type (optimizers {:stochastic-gradient-descent
                              {:nn-conf nn-conf
                               :step-fn (step-fns :gradient)
                               :listeners multiple-listeners
                               :termination-condition [(new-zero-direction-termination-condition)]
                               :model model}}))))
    (is (= org.deeplearning4j.optimize.solvers.StochasticGradientDescent
           (type (new-stochastic-gradient-descent-optimizer
                  :nn-conf nn-conf
                  :step-fn (step-fns :gradient)
                  :listeners multiple-listeners
                  :termination-condition [(new-zero-direction-termination-condition)]
                  :model model))))
    (is (= org.deeplearning4j.optimize.solvers.StochasticGradientDescent
           (type (optimizers {:stochastic-gradient-descent
                              {:nn-conf nn-conf
                               :step-fn (step-fns :gradient)
                               :listeners multiple-listeners
                               :model model}}))))
    (is (= org.deeplearning4j.optimize.solvers.StochasticGradientDescent
           (type (new-stochastic-gradient-descent-optimizer
                  :nn-conf nn-conf
                  :step-fn (step-fns :gradient)
                  :listeners multiple-listeners
                  :model model))))

    (is (= org.deeplearning4j.optimize.solvers.BackTrackLineSearch
           (type (optimizers {:back-track-line-search
                              {:model model
                               :optimizer (optimizers {:stochastic-gradient-descent
                                                       {:nn-conf nn-conf
                                                        :step-fn (step-fns :gradient)
                                                        :listeners multiple-listeners
                                                        :model model}})}}))))
    (is (= org.deeplearning4j.optimize.solvers.BackTrackLineSearch
           (type (new-back-track-line-search-optimizer
                  :model model
                  :optimizer (optimizers {:stochastic-gradient-descent
                                          {:nn-conf nn-conf
                                           :step-fn (step-fns :gradient)
                                           :listeners multiple-listeners
                                           :model model}})))))
    (is (= org.deeplearning4j.optimize.solvers.BackTrackLineSearch
           (type (optimizers {:back-track-line-search
                              {:layer model
                               :step-fn (step-fns :gradient)
                               :optimizer (optimizers {:stochastic-gradient-descent
                                                       {:nn-conf nn-conf
                                                        :step-fn (step-fns :gradient)
                                                        :listeners multiple-listeners
                                                        :model model}})}}))))
    (is (= org.deeplearning4j.optimize.solvers.BackTrackLineSearch
           (type (new-back-track-line-search-optimizer
                  :layer model
                  :step-fn (step-fns :gradient)
                  :optimizer (optimizers {:stochastic-gradient-descent
                                          {:nn-conf nn-conf
                                           :step-fn (step-fns :gradient)
                                           :listeners multiple-listeners
                                           :model model}})))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; API return type testing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; convex optimizer api
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/ConvexOptimizer.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest convex-optimizer-api
  (testing "the return type of the convex optimizer interace fns"
    (let [conj-grad-optim (new-conjugate-gradient-optimizer
                           :nn-conf nn-conf
                           :step-fn (step-fns :gradient)
                           :listeners multiple-listeners
                           :termination-condition [(new-zero-direction-termination-condition)]
                           :model model)
          lbfgs-optim (new-lbfgs-optimizer
                       :nn-conf nn-conf
                       :step-fn (step-fns :gradient)
                       :listeners multiple-listeners
                       :termination-condition [(new-zero-direction-termination-condition)]
                       :model model)
          line-grad-optim (new-line-gradient-descent-optimizer
                           :nn-conf nn-conf
                           :step-fn (step-fns :gradient)
                           :listeners multiple-listeners
                           :termination-condition [(new-zero-direction-termination-condition)]
                           :model model)
          stoch-grad-optim (new-stochastic-gradient-descent-optimizer
                            :nn-conf nn-conf
                            :step-fn (step-fns :gradient)
                            :listeners multiple-listeners
                            :termination-condition [(new-zero-direction-termination-condition)]
                            :model model)]
      ;; I believe this comes from the :input set in the creation of model
      (is (= java.lang.Integer (type (get-batch-size conj-grad-optim))))
      (is (= java.lang.Integer (type (get-batch-size lbfgs-optim))))
      (is (= java.lang.Integer (type (get-batch-size line-grad-optim))))
      (is (= java.lang.Integer (type (get-batch-size stoch-grad-optim))))
      ;; for the models used here, there was no input passed during creation
      (is (= 0 (get-batch-size stoch-grad-optim)))
      (is (= 0 (get-batch-size lbfgs-optim)))
      (is (= 0 (get-batch-size line-grad-optim)))
      (is (= 0 (get-batch-size stoch-grad-optim)))

      ;; determined by the termination-condition passed in the creation of the optimizer
      (is (= java.lang.Boolean (type (check-terminal-conditions?
                                      :optim conj-grad-optim
                                      :gradient (row-vector [1 2 3 4 5])
                                      :old-score 10.2
                                      :score 5.5
                                      :iteration 2))))
      (is (= java.lang.Boolean (type (check-terminal-conditions?
                                      :optim lbfgs-optim
                                      :gradient (row-vector [1 2 3 4 5])
                                      :old-score 10.2
                                      :score 5.5
                                      :iteration 2))))
      (is (= java.lang.Boolean (type (check-terminal-conditions?
                                      :optim line-grad-optim
                                      :gradient (row-vector [1 2 3 4 5])
                                      :old-score 10.2
                                      :score 5.5
                                      :iteration 2))))
      (is (= java.lang.Boolean (type (check-terminal-conditions?
                                      :optim stoch-grad-optim
                                      :gradient (row-vector [1 2 3 4 5])
                                      :old-score 10.2
                                      :score 5.5
                                      :iteration 2))))

      ;; comes from the nn-conf passed in the creation of the optimizer
      (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
             (type (get-conf conj-grad-optim))))
      (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
             (type (get-conf lbfgs-optim))))
      (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
             (type (get-conf line-grad-optim))))
      (is (= org.deeplearning4j.nn.conf.NeuralNetConfiguration
             (type (get-conf stoch-grad-optim))))

      ;; comes from the model passed in the creation of the optimizer
      ;; (model =  a layer)
      (is (= org.deeplearning4j.nn.updater.LayerUpdater
             (type (get-updater conj-grad-optim))))
      (is (= org.deeplearning4j.nn.updater.LayerUpdater
             (type (get-updater lbfgs-optim))))
      (is (= org.deeplearning4j.nn.updater.LayerUpdater
             (type (get-updater line-grad-optim))))
      (is (= org.deeplearning4j.nn.updater.LayerUpdater
             (type (get-updater stoch-grad-optim))))

      ;; need to figure out how to test return type of get-gradient-and-score
      ;; believe I need to use a trained model here instead of an untrained layer
      ;; ^ this also applies to optimize,

      ;; not sure what line is suppose to be or when it needs to be used
      ;; *************************************************************
      ;; hence why two of the tests are commented out...
      #_(is (= (type conj-grad-optim)
             (type (post-step! :optim conj-grad-optim
                               :line (row-vector [1 2 3 4 5])))))
      #_(is (= (type lbfgs-optim)
             (type (post-step! :optim lbfgs-optim
                               :line (row-vector [1 2 3 4 5])))))
      (is (= (type line-grad-optim)
             (type (post-step! :optim line-grad-optim
                               :line (row-vector [1 2 3 4 5])))))
      (is (= (type stoch-grad-optim)
             (type (post-step! :optim stoch-grad-optim
                               :line (row-vector [1 2 3 4 5])))))

      ;; not sure what this method is doing behind the scenes
      ;; *************************************************************
      ;; hence why two of the tests are commented out...
      #_(is (= (type conj-grad-optim)  (type (pre-process-line! conj-grad-optim))))
      #_(is (= (type lbfgs-optim)  (type (pre-process-line! lbfgs-optim))))
      (is (= (type line-grad-optim)  (type (pre-process-line! line-grad-optim))))
      (is (= (type stoch-grad-optim)  (type (pre-process-line! stoch-grad-optim))))

      ;; get the score from the model
      (is (= java.lang.Double (type (get-score conj-grad-optim))))
      (is (= java.lang.Double (type (get-score lbfgs-optim))))
      (is (= java.lang.Double (type (get-score line-grad-optim))))
      (is (= java.lang.Double (type (get-score stoch-grad-optim))))

      ;; testing setting the batch size
      (is (= 5 (get-batch-size (set-batch-size! :optim conj-grad-optim
                                                :batch-size 5))))
      (is (= 5 (get-batch-size (set-batch-size! :optim lbfgs-optim
                                                :batch-size 5))))
      (is (= 5 (get-batch-size (set-batch-size! :optim line-grad-optim
                                                :batch-size 5))))
      (is (= 5 (get-batch-size (set-batch-size! :optim stoch-grad-optim
                                                :batch-size 5))))

      ;; ensuring the optimizer is returned after setting the listeners
      (is (= (type conj-grad-optim)
             (type (set-listeners! :optim conj-grad-optim
                                   :listeners multiple-listeners))))
      (is (= (type lbfgs-optim)
             (type (set-listeners! :optim lbfgs-optim
                                   :listeners multiple-listeners))))
      (is (= (type line-grad-optim)
             (type (set-listeners! :optim line-grad-optim
                                   :listeners multiple-listeners))))
      (is (= (type stoch-grad-optim)
             (type (set-listeners! :optim stoch-grad-optim
                                   :listeners multiple-listeners))))

      ;; making sure we get back a layer updater after setting it
      (is (= (type (new-layer-updater))
             (type (get-updater (set-updater! :optim conj-grad-optim
                                              :updater (new-layer-updater))))))
      (is (= (type (new-layer-updater))
             (type (get-updater (set-updater! :optim lbfgs-optim
                                              :updater (new-layer-updater))))))
      (is (= (type (new-layer-updater))
             (type (get-updater (set-updater! :optim line-grad-optim
                                              :updater (new-layer-updater))))))
      (is (= (type (new-layer-updater))
             (type (get-updater (set-updater! :optim stoch-grad-optim
                                              :updater (new-layer-updater))))))

      ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
      ;; come back to set-up-search-state!, need to implement the berkeley pairs
      ;; but need to test anyways
      #_(is (= "" (set-up-search-state! :optim conj-grad-optim
                                      :gradient (constructor)
                                      :score 10)))
      ;; ensure the optimizer is returned
      (is (= (type conj-grad-optim)
             (type (update-gradient-according-to-params! :optim conj-grad-optim
                                                         :gradient (new-default-gradient)
                                                         :model model
                                                         :batch-size 5))))
      (is (= (type lbfgs-optim)
             (type (update-gradient-according-to-params! :optim lbfgs-optim
                                                         :gradient (new-default-gradient)
                                                         :model model
                                                         :batch-size 5))))
      (is (= (type line-grad-optim)
             (type (update-gradient-according-to-params! :optim line-grad-optim
                                                         :gradient (new-default-gradient)
                                                         :model model
                                                         :batch-size 5))))
      (is (= (type stoch-grad-optim)
             (type (update-gradient-according-to-params! :optim stoch-grad-optim
                                                         :gradient (new-default-gradient)
                                                         :model model
                                                         :batch-size 5)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Iteration Listener api return type testing
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/IterationListener.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest iteration-listener-test
  (testing "the iteration listener api/interface return types"
    (let [p-and-g (new-param-and-gradient-iteration-listener)
          coll-scores (new-collection-scores-iteration-listener :frequency 5)
          composed (new-composable-iteration-listener :coll-of-listeners multiple-listeners)
          score-iter (new-score-iteration-listener :print-every-n 5)
          perf (new-performance-iteration-listener)]
      ;; make sure invoke! returns the listener
      (is (= (type p-and-g) (type (invoke! p-and-g))))
      (is (= (type coll-scores) (type (invoke! coll-scores))))
      (is (= (type composed) (type (invoke! composed))))
      (is (= (type score-iter) (type (invoke! score-iter))))
      (is (= (type perf) (type (invoke! perf))))

      ;; checking that the return type of invoked? is a boolean
      (is (= java.lang.Boolean (type (invoked? p-and-g))))
      (is (= java.lang.Boolean (type (invoked? coll-scores))))
      (is (= java.lang.Boolean (type (invoked? composed))))
      (is (= java.lang.Boolean (type (invoked? score-iter))))
      (is (= java.lang.Boolean (type (invoked? perf)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Step Function api return type testing
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/StepFunction.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest step-api
  (testing "the step api"
    (let [gradient-step (step-fns :gradient)
          neg-gradient-step (step-fns :negative-gradient)]
      ;; only supplying the step fn
      (is (= (type gradient-step) (type (step! :step-fn gradient-step))))
      (is (= (type neg-gradient-step) (type (step! :step-fn neg-gradient-step))))

      ;; supplying the step fn, features and the line
      (is (= (type gradient-step) (type (step! :step-fn gradient-step
                                               :features (rand [2 2])
                                               :line (rand [2 2])))))
      (is (= (type neg-gradient-step) (type (step! :step-fn neg-gradient-step
                                               :features (rand [2 2])
                                               :line (rand [2 2])))))

      ;; supplying the step fn, features, line and step value
      (is (= (type gradient-step) (type (step! :step-fn gradient-step
                                               :features (rand [2 2])
                                               :line (rand [2 2])
                                               :step 1.0))))
      (is (= (type neg-gradient-step) (type (step! :step-fn neg-gradient-step
                                                   :features (rand [2 2])
                                                   :line (rand [2 2])
                                                   :step 1.0)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; termination condition api return type testing
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/TerminationCondition.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest termination-cond-test
  (testing "the return type of terminate from the termation condition interface"
    (let [eps (termination-condition {:eps {:eps 2.0 :tolerance 5.0}})
          norm-2 (termination-condition {:norm-2 {:gradient-tolerance 5.0}})
          zero-dir (termination-condition {:zero-direction {}})
          param  (array-of :java-type java.lang.Object
                           :data (rand [1]))]
      (is (= java.lang.Boolean (type (terminate? :term-cond eps
                                                 :cost 2.0
                                                 :old-cost 5.0
                                                 :other-params param))))
      (is (= java.lang.Boolean (type (terminate? :term-cond norm-2
                                                 :cost 2.0
                                                 :old-cost 5.0
                                                 :other-params param))))
      (is (= java.lang.Boolean (type (terminate? :term-cond zero-dir
                                                 :cost 2.0
                                                 :old-cost 5.0
                                                 :other-params param)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; training listener interface, implementing classes not wrapped yet (ui)
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/TrainingListener.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; making a note for future me


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; misc fns that are not from a class instead of an interface
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest misc-methods
  (testing "the methods defined within classes instead of interfaces"
    (let [l (new-collection-scores-iteration-listener :frequency 5)
          csv (io/as-file "resources/test.csv")
          tabs (io/as-file "resources/secondtest.txt")
          back-track (new-back-track-line-search-optimizer
                      :model model
                      :optimizer (optimizers {:stochastic-gradient-descent
                                              {:nn-conf nn-conf
                                               :step-fn (step-fns :gradient)
                                               :listeners multiple-listeners
                                               :model model}}))]
      ;; export-scores! to a file
      ;; found in dl4clj.optimize.listeners.listeners
      (is (= org.deeplearning4j.optimize.listeners.CollectScoresIterationListener
             (type (export-scores! :listener l
                                   :file csv
                                   :delim ","))))
      (is (= "Iteration,Score" (slurp "resources/test.csv")))
      (is (= org.deeplearning4j.optimize.listeners.CollectScoresIterationListener
             (type (export-scores! :listener l
                                   :file tabs))))
      (is (= "Iteration\tScore" (slurp "resources/secondtest.txt")))

      ;; export-scores! to an output stream
      ;; found in dl4clj.optimize.listeners.listeners
      (with-open [o (io/output-stream "resources/output-stream.csv")
                  o-no-delim (io/output-stream "resources/output-s.txt")]
        (is (= org.deeplearning4j.optimize.listeners.CollectScoresIterationListener
               (type (export-scores! :listener l
                                     :output-stream o
                                     :delim ","))))
        (is (= org.deeplearning4j.optimize.listeners.CollectScoresIterationListener
               (type (export-scores! :listener l
                                     :output-stream o-no-delim)))))
      (is (= "Iteration,Score" (slurp "resources/output-stream.csv")))
      (is (= "Iteration\tScore" (slurp "resources/output-s.txt")))

      ;; get-optimizer found in dl4clj.optimize.solver
      (is (= org.deeplearning4j.optimize.solvers.StochasticGradientDescent
             (type (get-optimizer (build-solver
                                   :nn-conf nn-conf
                                   :model model
                                   :single-listener single-listener)))))

      ;; get-default-step-fn-for-optimizer in  dl4clj.optimize.solvers.optimizers
      (is (= org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction
             (type (get-default-step-fn-for-optimizer
                    (type (new-conjugate-gradient-optimizer
                           :nn-conf nn-conf
                           :step-fn (step-fns :gradient)
                           :listeners multiple-listeners
                           :termination-condition [(new-zero-direction-termination-condition)]
                           :model model))))))
      (is (= org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction
             (type (get-default-step-fn-for-optimizer
                    (type (new-lbfgs-optimizer
                           :nn-conf nn-conf
                           :step-fn (step-fns :gradient)
                           :listeners multiple-listeners
                           :termination-condition [(new-zero-direction-termination-condition)]
                           :model model))))))
      (is (= org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction
             (type (get-default-step-fn-for-optimizer
                    (type (new-line-gradient-descent-optimizer
                           :nn-conf nn-conf
                           :step-fn (step-fns :gradient)
                           :listeners multiple-listeners
                           :termination-condition [(new-zero-direction-termination-condition)]
                           :model model))))))
      (is (= org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction
             (type (get-default-step-fn-for-optimizer
                    (type (new-stochastic-gradient-descent-optimizer
                           :nn-conf nn-conf
                           :step-fn (step-fns :gradient)
                           :listeners multiple-listeners
                           :termination-condition [(new-zero-direction-termination-condition)]
                           :model model))))))

      ;; get-iteration-count in dl4clj.optimize.solvers.optimizers
      (is (= java.lang.Integer (type (get-iteration-count model))))

      ;; increment-iteration-count! in dl4clj.optimize.solvers.optimizers
      ;; look at the model itself to see that the iteration count has increased
      (is (= (type model) (type (increment-iteration-count! :model model
                                                            :increment-by 1))))

      ;; back track line search optimizer methods
      ;; see the bottom of dl4clj.optimize.solvers.optimizers
      (is (= java.lang.Integer (type (get-max-iterations back-track))))
      (is (= java.lang.Double (type (get-step-max back-track))))
      (is (= (type back-track) (type
                                (set-abs-tolerance! :back-track back-track
                                                    :tolerance 0.2))))
      (is (= (type back-track) (type
                                (set-max-iterations! :back-track back-track
                                                     :max-iterations 120))))
      (is (= (type back-track) (type
                                (set-relative-tolerance! :back-track back-track
                                                         :tolerance 0.1))))
      ;; need to figure out what to pass when methods wants params as an INDArray
      ;; I always get this error: Unable to set parameters: must be of length 0
      ;; I think I need to make an empty INDArray but cant with the current
      ;; ND4j INDArray-creation ns
      #_(is (= (type back-track) (type
                                (set-score-for! :back-track back-track
                                                :params (zeros 1)))))
      (is (= (type back-track) (type
                                (set-step-max! :back-track back-track
                                               :step-max 1.0)))))))
