(ns dl4clj.optimize-tests
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
            [dl4clj.nn.layers.recurrent.graves-lstm :refer [new-graves-lstm-layer]]
            [nd4clj.linalg.factory.nd4j :refer :all]
            [dl4clj.nn.layers.variational-autoencoder :refer [new-variational-autoencoder]])
  (:import [org.deeplearning4j.optimize.api IterationListener]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; objects that I need for testing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def nn-conf
  (nn/nn-conf-builder
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
             :dist {:uniform {:lower -0.08, :upper 0.08}}}}}))

(def nn-conf-unsupervised
  (nn/nn-conf-builder
   {:seed 123
    :iterations 1
    :optimization-algo :stochastic-gradient-descent
    :learning-rate 1e-2
    :updater :rmsprop
    :rms-decay 0.95
    :weight-init :xavier
    :regularization true
    :l2 1e-4
    :layer {:variational-auto-encoder {:activation-fn :leaky-relu
                                       :encoder-layer-sizes [256 256]
                                       :decoder-layer-sizes [256 256]
                                       :pzx-activation-function :identity
                                       :reconstruction-distribution {:bernoulli {:activation-fn :sigmoid}}
                                       :n-in 5
                                       :n-out 2}}}))

(def single-listener (new-score-iteration-listener :print-every-n 2 :array? true))

(def multiple-listeners [(new-score-iteration-listener :print-every-n 2)
                         (new-collection-scores-iteration-listener :frequency 2)])

(def model (new-graves-lstm-layer :conf nn-conf))

(def unsupervised-model (new-variational-autoencoder :conf nn-conf-unsupervised))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of generic solvers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest generic-solvers
  ;; still need to write a good test for optimize!
  ;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/package-summary.html
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
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest step-functions
  (testing "the creation of step functions and that the fns implement the step function interface"
    ;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/stepfunctions/package-summary.html
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

    (is (= org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction
           (type (step! :step-fn (step-fns :negative-gradient)))))
    ;; will need to expand step! test to include the INDArray args
    ))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of listeners
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest listeners-test
  (testing "the creation of iteration listeners"
    ;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/listeners/package-summary.html
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
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest terminations
  (testing "the creation of termination conditions"
    ;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/terminations/package-summary.html
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
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest various-optimizers
  (testing "the creation of optimizers"
    ;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/solvers/package-summary.html
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

(deftest convex-optimizer-api
  (testing "the return type of the convex optimizer interace fns"
    ;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/ConvexOptimizer.html
    (let [conj-grad-optim (new-conjugate-gradient-optimizer
                           :nn-conf nn-conf
                           :step-fn (step-fns :gradient)
                           :listeners multiple-listeners
                           :termination-condition [(new-zero-direction-termination-condition)]
                           :model model)
          unsupervised-optim (new-stochastic-gradient-descent-optimizer
                              :nn-conf nn-conf-unsupervised
                              :step-fn (step-fns :gradient)
                              :listeners multiple-listeners
                              :termination-condition [(new-zero-direction-termination-condition)]
                              :model unsupervised-model)
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


      )
    ))

;; unique to optimize.Solver
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/Solver.html
#_(is (= org.deeplearning4j.optimize.solvers.StochasticGradientDescent
       (type (get-optimizer (build-solver
                             :nn-conf nn-conf
                             :model model
                             :single-listener single-listener)))))
