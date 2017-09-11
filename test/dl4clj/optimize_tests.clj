(ns dl4clj.optimize-tests
  (:require [dl4clj.optimize.listeners :refer :all]
            [dl4clj.optimize.api.listeners :refer :all]
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
