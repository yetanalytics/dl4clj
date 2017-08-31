(ns dl4clj.optimize-tests
  (:require [dl4clj.optimize.listeners :refer :all]
            [dl4clj.optimize.api.iteration-listener :refer :all]
            [dl4clj.optimize.api.listeners :refer :all]
            [dl4clj.optimize.api.training-listener :refer :all]
            [clojure.test :refer :all])
  (:import [org.deeplearning4j.optimize.api IterationListener]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; objects that I need for testing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def single-listener (new-score-iteration-listener :print-every-n 2 :array? true))

(def multiple-listeners [(new-score-iteration-listener :print-every-n 2)
                         (new-collection-scores-iteration-listener :frequency 2)])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of listeners
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/listeners/package-summary.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest listeners-test
  (testing "the creation of iteration listeners"
    (is (= org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener
           (type (new-param-and-gradient-iteration-listener :as-code? false))))
    (is (= org.deeplearning4j.optimize.listeners.CollectScoresIterationListener
           (type (new-collection-scores-iteration-listener :frequency 5 :as-code? false))))
    (is (= org.deeplearning4j.optimize.listeners.CollectScoresIterationListener
           (type (new-collection-scores-iteration-listener :as-code? false))))
    (is (= org.deeplearning4j.optimize.listeners.ComposableIterationListener
           (type (new-composable-iteration-listener :coll-of-listeners multiple-listeners :as-code? false))))
    (is (= org.deeplearning4j.optimize.listeners.ScoreIterationListener
           (type (new-score-iteration-listener :print-every-n 5 :as-code? false))))
    (is (= org.deeplearning4j.optimize.listeners.ScoreIterationListener
           (type (new-score-iteration-listener :as-code? false))))
    (is (= org.deeplearning4j.optimize.listeners.PerformanceListener
           (type (new-performance-iteration-listener :as-code? false :build? true))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Iteration Listener api return type testing
;; https://deeplearning4j.org/doc/org/deeplearning4j/optimize/api/IterationListener.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest iteration-listener-test
  (testing "the iteration listener api/interface return types"
    (let [p-and-g (new-param-and-gradient-iteration-listener)
          coll-scores (new-collection-scores-iteration-listener :frequency 5 :as-code? false)
          composed (new-composable-iteration-listener :coll-of-listeners multiple-listeners :as-code? false)
          score-iter (new-score-iteration-listener :print-every-n 5 :as-code? false)
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
