(ns dl4clj.eval-test
  (:require [dl4clj.eval.confusion-matrix :refer :all]
            [clojure.test :refer :all]
            [dl4clj.eval.eval-tools :refer :all]
            [dl4clj.eval.eval-utils :refer :all]
            [dl4clj.eval.evaluation :refer :all]
            [dl4clj.eval.interface.i-evaluation :refer :all]
            [dl4clj.eval.roc.rocs :refer :all]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; objects that I need for testing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; need to set up the mnist example and do some early stopping training (control training time)
;; then use that for testing
;; don't think that will work for binary-rocs, so i will have to look into dl4j unit tests for rocs
;; to find a default dataset to use

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of evaluators and regression evaluators
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/Evaluation.html
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/RegressionEvaluation.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest rocs-test
  (testing "the creation and interaction with binary and multi-class rocs"
    (is (= org.deeplearning4j.eval.ROC (type (new-binary-roc :threshold-steps 2))))
    (is (= org.deeplearning4j.eval.ROCMultiClass (type (new-multiclass-roc :threshold-steps 2))))

    ;; this has no data in it so it returns NaN
    (is (= java.lang.Double
           (type (calculate-area-under-curve :roc (new-binary-roc :threshold-steps 2)))))
    ;; need to get data into this before i can get its auc
    #_(is (= java.lang.Double
           (type (calculate-area-under-curve :roc (new-multiclass-roc :threshold-steps 2)
                                             :class-idx 1))))

    ))
