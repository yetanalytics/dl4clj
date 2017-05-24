(ns dl4clj.eval-test
  (:require [dl4clj.eval.confusion-matrix :refer :all]
            [clojure.test :refer :all]
            [dl4clj.eval.eval-tools :refer :all]
            [dl4clj.eval.eval-utils :refer :all]
            [dl4clj.eval.evaluation :refer :all]
            [dl4clj.eval.interface.i-evaluation :refer :all]
            [dl4clj.eval.roc.rocs :refer :all]
            ;; requireing early stopping ns for minimal training
            [dl4clj.datasets.iterator.impl.default-datasets :refer [new-mnist-data-set-iterator
                                                                    next-data-point
                                                                    get-feature-matrix]]
            [nd4clj.linalg.api.ds-iter :refer [get-labels reset-iter!]]
            [dl4clj.earlystopping.early-stopping-trainer :refer :all]
            [dl4clj.earlystopping.termination-conditions :refer :all]
            [dl4clj.earlystopping.early-stopping-config :refer :all]
            [dl4clj.earlystopping.early-stopping-result :refer :all]
            [dl4clj.earlystopping.model-saver :refer :all]
            [dl4clj.earlystopping.score-calc :refer :all]
            [dl4clj.earlystopping.interfaces.early-stopping-trainer :refer [fit-trainer!]]
            [dl4clj.nn.conf.builders.nn-conf-builder :as nn]
            [dl4clj.nn.multilayer.multi-layer-network :as ml]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; objects that I need for testing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;; fully train a network, save it and then import it here for testing
;; this set up doesn't produce any predictions bc of low training
(def mln (ml/multi-layer-network
          (nn/nn-conf-builder
           {:seed 123
            :optimization-algo :stochastic-gradient-descent
            :iterations 1
            :learning-rate 0.006
            :updater :nesterovs
            :momentum 0.9
            :regularization true
            :l2 1e-4
            :layers {0 {:dense-layer {:n-in 784
                                      :n-out 1000
                                      :activation-fn :relu
                                      :weight-init :xavier}}
                     1 {:output-layer {:loss-fn :negativeloglikelihood
                                       :n-in 1000
                                       :n-out 10
                                       :activation-fn :soft-max
                                       :weight-init :xavier}}}})))

(def mnist-train (new-mnist-data-set-iterator :batch-size 25 :train? true
                                              :seed 123 :n-examples 1024
                                              :binarize? false :shuffle? true))

(def mnist-test (new-mnist-data-set-iterator :batch-size 25 :train? false
                                             :seed 123 :n-examples 512
                                             :binarize? false :shuffle? true))

(def es-conf (new-early-stopping-config
              :epoch-termination-conditions (new-max-epochs-termination-condition 1)
              :iteration-termination-conditions (new-max-time-iteration-termination-condition
                                                 :max-time-val 60
                                                 :max-time-unit :seconds)
              :n-epochs 1
              :model-saver (new-in-memory-saver)
              :save-last-model? true
              :score-calculator (new-data-set-loss-calculator
                                 :ds-iter mnist-test
                                 :average? true)))


(def es-trained (get-best-model-from-result
                 (fit-trainer! (new-early-stopping-trainer
                                :early-stopping-conf es-conf
                                :mln mln
                                :training-dataset-iterator mnist-train))))
;; don't think that will work for binary-rocs, so i will have to look into dl4j unit tests for rocs
;; to find a default dataset to use

;; also going to need a timeseries trained network
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of evaluators and rocs
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/Evaluation.html
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/RegressionEvaluation.html
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROC.html
;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/ROCMultiClass.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest evaler-creation-test
  (testing "the creation of evaluators"
    ;; classification evaluation
    (is (= org.deeplearning4j.eval.Evaluation (type (new-classification-evaler))))
    (is (= org.deeplearning4j.eval.Evaluation (type (new-classification-evaler
                                                     :n-classes 5))))
    (is (= org.deeplearning4j.eval.Evaluation (type (new-classification-evaler
                                                     :label-to-idx {0 "foo" 1 "baz"}))))
    (is (= org.deeplearning4j.eval.Evaluation (type (new-classification-evaler
                                                     :labels ["foo" "baz"]))))
    (is (= org.deeplearning4j.eval.Evaluation (type (new-classification-evaler
                                                     :labels '("foo" "baz")))))
    (is (= org.deeplearning4j.eval.Evaluation (type (new-classification-evaler
                                                     :labels ["foo" "baz"]
                                                     :top-n 2))))

    ;; regression/timeseries evaluation
    (is (= org.deeplearning4j.eval.RegressionEvaluation
           (type (new-regression-evaler :n-columns 2))))
    (is (= org.deeplearning4j.eval.RegressionEvaluation
           (type (new-regression-evaler :column-names ["foo" "baz"]))))
    (is (= org.deeplearning4j.eval.RegressionEvaluation
           (type (new-regression-evaler :column-names (list "foo" "baz")))))
    (is (= org.deeplearning4j.eval.RegressionEvaluation
           (type (new-regression-evaler :n-columns 2 :precision 2))))
    (is (= org.deeplearning4j.eval.RegressionEvaluation
           (type (new-regression-evaler :column-names ["foo" "baz"]
                                        :precision 1))))

    ;; ROC evaluation
    (is (= org.deeplearning4j.eval.ROC (type (new-binary-roc :threshold-steps 2))))
    (is (= org.deeplearning4j.eval.ROCMultiClass (type (new-multiclass-roc :threshold-steps 2))))

    ;; eval utils
    (is (= org.deeplearning4j.eval.EvaluationUtils (type (new-evaluation-utils))))
    ))

(deftest eval-classification-with-data
  (testing "the use of classification evalers"
    (let [data (next-data-point (reset-iter! mnist-test))
          features (get-feature-matrix data)
          mln-output (ml/output :mln es-trained :input features)
          evalr (new-classification-evaler)
          labels (get-labels data)
          evaler-with-data (eval-classification! :evaler evalr :features features
                                                 :mln es-trained :labels labels)]
      (is (= java.lang.Double (type (get-accuracy evaler-with-data))))
      (is (= (type evalr) (type
                           (add-to-confusion
                            :evaler evalr :real-value 2 :guess-value 1))))
      (is (= java.lang.Integer (type
                                (class-count :evaler evaler-with-data
                                             :class-label-idx 0))))
      (is (= java.lang.String (type (confusion-to-string evaler-with-data))))
      (is (= java.lang.Double (type (f1 :evaler evaler-with-data :class-label-idx 0))))
      (is (= java.lang.Double (type (f1 :evaler evaler-with-data))))
      (is (= java.lang.Double (type (false-alarm-rate evaler-with-data))))
      (is (= java.lang.Double (type (false-negative-rate :evaler evaler-with-data))))
      (is (= java.lang.Double (type
                               (false-negative-rate :evaler evaler-with-data
                                                    :class-label-idx 0))))
      (is (= java.lang.Double (type
                               (false-negative-rate :evaler evaler-with-data
                                                    :class-label-idx 0
                                                    :edge-case 0.2))))
      (is (= java.util.HashMap (type (false-negatives evaler-with-data))))
      (is (= java.lang.Double (type (false-positive-rate :evaler evaler-with-data))))
      (is (= java.lang.Double (type
                               (false-positive-rate :evaler evaler-with-data
                                                    :class-label-idx 1))))
      (is (= java.lang.Double (type
                               (false-positive-rate :evaler evaler-with-data
                                                    :class-label-idx 1
                                                    :edge-case 0.3))))
      (is (= java.util.HashMap (type (false-positives evaler-with-data))))
      (is (= java.lang.String (type
                               (get-class-label :evaler evaler-with-data
                                                :label-idx 1))))
      (is (= org.deeplearning4j.eval.ConfusionMatrix
             (type (get-confusion-matrix evaler-with-data))))
      (is (= java.lang.Integer (type (get-num-row-counter evaler-with-data))))
      ;; the network can't make any predictions bc of the low amount of training?
      ;; will need to import a trained network and test this theory
      ;; could also be nill because of aproblem creating the prediction objects
      ;; https://deeplearning4j.org/doc/org/deeplearning4j/eval/meta/Prediction.html
      ;; could also be because the threshold was set too high
      ;; try changing this and see if these are no longer nil
      #_(is (= java.util.List (type (get-prediction-by-predicted-class
                                   :evaler evaler-with-data
                                   :idx-of-predicted-class 2))))
      #_(is (= java.util.List (type (get-prediction-errors evaler-with-data))))
      #_(is (= java.util.List (type (get-predictions :evaler evaler-with-data
                                                   :actual-class-idx 0
                                                   :predicted-class-idx 1))))
      #_(is (= java.util.List (type (get-predictions-by-actual-class :evaler evaler-with-data
                                                       :actual-class-idx 0))))
      (is (= java.lang.Integer (type (get-top-n-correct-count evaler-with-data))))
      (is (= java.lang.Integer (type (get-top-n-total-count evaler-with-data))))
      (is (= org.deeplearning4j.eval.Evaluation (type
                                                 (increment-false-negatives!
                                                  :evaler evalr :class-label-idx 0))))
      (is (= org.deeplearning4j.eval.Evaluation (type
                                                 (increment-false-positives!
                                                  :evaler evalr :class-label-idx 0))))
      (is (= org.deeplearning4j.eval.Evaluation (type
                                                 (increment-true-negatives!
                                                  :evaler evalr :class-label-idx 0))))
      (is (= org.deeplearning4j.eval.Evaluation (type
                                                 (increment-true-positives!
                                                  :evaler evalr :class-label-idx 0))))
      (is (= java.util.HashMap (type (total-negatives evaler-with-data))))
      (is (= java.util.HashMap (type (total-positives evaler-with-data))))
      (is (= java.lang.Double (type (get-precision :evaler evaler-with-data))))
      (is (= java.lang.Double (type
                               (get-precision :evaler evaler-with-data
                                              :class-label-idx 0))))
      (is (= java.lang.Double (type
                               (get-precision :evaler evaler-with-data
                                              :class-label-idx 0
                                              :edge-case 0.1))))
      (is (= java.lang.Double (type (recall :evaler evaler-with-data))))
      (is (= java.lang.Double (type
                               (recall
                                :evaler evaler-with-data
                                :class-label-idx 0))))
      (is (= java.lang.Double (type
                               (recall
                                :evaler evaler-with-data
                                :class-label-idx 0
                                :edge-case 0.1))))
      (is (= java.lang.Double (type (top-n-accuracy evaler-with-data))))
      (is (= java.util.HashMap (type (true-negatives evaler-with-data))))
      (is (= java.util.HashMap (type (true-positives evaler-with-data))))
      (is (= java.lang.String (type (get-stats :evaler evaler-with-data))))
      (is (= java.lang.String (type
                               (get-stats :evaler evaler-with-data
                                          :suppress-warnings? false))))
      (is (= org.deeplearning4j.eval.Evaluation (type (eval-classification!
                                                       :evaler evalr :predicted-idx 0
                                                       :actual-idx 1))))
      (is (= org.deeplearning4j.eval.Evaluation (type (eval-classification!
                                                       :evaler evalr :labels labels
                                                       :network-predictions mln-output))))
      ;; need to come back and test with a mask-array and record-meta-data
      ;; want to use real examples of those to vars tho instead of a rand INDArray

      ;; need a time series network for eval-time-series!

      (is (= org.deeplearning4j.eval.Evaluation (type
                                                 (merge! :evaler evaler-with-data
                                                         :other-evaler evalr))))
      )
    ))

(deftest confusion-matrix-test
  (testing "the creation and manipulation of confusion matrices"
    (let [data (next-data-point (reset-iter! mnist-test))
          features (get-feature-matrix data)
          evalr (new-classification-evaler)
          labels (get-labels data)
          evaler-with-data (eval-classification! :evaler evalr :features features
                                                 :mln es-trained :labels labels)
          confusion (get-confusion-matrix evaler-with-data)
          other-confusion (new-confusion-matrix :existing-confusion-matrix confusion)]
      (is (= org.deeplearning4j.eval.ConfusionMatrix
             (type (new-confusion-matrix
                    :existing-confusion-matrix confusion))))
      (is (= org.deeplearning4j.eval.ConfusionMatrix
             (type (new-confusion-matrix
                    :classes [java.lang.Double java.lang.Double]))))
      (is (= org.deeplearning4j.eval.ConfusionMatrix
             (type
              (add! :base-confusion-matrix confusion
                    :other-confusion-matrix other-confusion))))
      (is (= org.deeplearning4j.eval.ConfusionMatrix
             (type
              (add! :base-confusion-matrix confusion
                    :actual 2.0
                    :predicted 1.0))))
      (is (= org.deeplearning4j.eval.ConfusionMatrix
             (type
              (add! :base-confusion-matrix confusion
                    :actual 2.0
                    :predicted 1.0
                    :n 2))))
      (is (= java.lang.Integer (type
                                (get-actual-total :confusion-matrix confusion
                                                  :actual 1))))
      (is (= java.util.ArrayList (type (get-classes confusion))))
      (is (= java.lang.Integer (type
                                (get-count :confusion-matrix confusion
                                           :actual 1 :predicted 2))))
      (is (= java.lang.Integer (type
                                (get-predicted-total :confusion-matrix confusion
                                                     :predicted 1))))
      (is (= java.lang.String (type (to-csv confusion))))
      (is (= java.lang.String (type (to-html confusion)))))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; left to test
;; time-series classification
;; time-series regression
;; eval utils (timeseries)
;; binary roc
;; multi-class roc
;; eval-tools (roc)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#_(deftest rocs-test
  (testing "the creation and interaction with binary and multi-class rocs"
    ;; this has no data in it so it returns NaN
    (is (= java.lang.Double
           (type (calculate-area-under-curve :roc (new-binary-roc :threshold-steps 2)))))
    ;; need to get data into this before i can get its auc
    #_(is (= java.lang.Double
             (type (calculate-area-under-curve :roc (new-multiclass-roc :threshold-steps 2)
                                               :class-idx 1))))
    ))
