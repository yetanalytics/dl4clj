(ns dl4clj.eval-test
  (:require [dl4clj.eval.confusion-matrix :refer :all]
            [clojure.test :refer :all]
            [dl4clj.eval.eval-tools :refer :all]
            [dl4clj.eval.eval-utils :refer :all]
            [dl4clj.eval.evaluation :refer :all]
            [dl4clj.eval.roc :refer :all]
            [dl4clj.eval.roc-multi-class :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; objects that I need for testing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of evaluators and regression evaluators
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/Evaluation.html
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/RegressionEvaluation.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
