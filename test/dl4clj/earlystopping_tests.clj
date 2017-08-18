(ns dl4clj.earlystopping-tests
  (:require [dl4clj.earlystopping.early-stopping-config :refer [new-early-stopping-config]]
            [dl4clj.earlystopping.early-stopping-result :refer [new-early-stopping-result]]
            [dl4clj.earlystopping.early-stopping-trainer :refer [new-early-stopping-trainer]]
            #_[dl4clj.earlystopping.api.early-stopping-trainer :refer :all]
            #_[dl4clj.earlystopping.model-saver :refer [new-in-memory-saver new-local-file-model-saver]]
            #_[dl4clj.earlystopping.api.model-saver :refer :all]
            #_[dl4clj.earlystopping.score-calc :refer [new-ds-loss-calculator]]
            #_[dl4clj.earlystopping.api.score-calc :refer :all]
            #_[dl4clj.earlystopping.termination-conditions :refer :all]
            #_[dl4clj.earlystopping.api.epoch-termination-condition :refer :all]
            #_[dl4clj.earlystopping.api.iteration-termination-condition :refer :all]
            #_[dl4clj.earlystopping.api.listener :refer :all]
            ;; namespaces I need to test the above namespaces
            [dl4clj.nn.multilayer.multi-layer-network :refer [new-multi-layer-network]]
            [dl4clj.nn.conf.builders.nn :as nn]
            [dl4clj.nn.api.model :refer [init! score!]]
            #_[dl4clj.datasets.iterators :refer [new-record-reader-dataset-iterator
                                               new-mnist-data-set-iterator]]
            [dl4clj.datasets.record-readers :refer [new-csv-record-reader]]
            [dl4clj.datasets.input-splits :refer [new-filesplit]]
            #_[dl4clj.datasets.record-readers :refer [new-csv-record-reader]]
            [dl4clj.datasets.api.record-readers :refer [initialize-rr!]]
            #_[dl4clj.datasets.default-datasets :refer [new-mnist-ds]]
            [clojure.test :refer :all])
  (:import [java.nio.charset Charset]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; objs needed in multiple tests
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def mln-code
  (new-multi-layer-network
   :as-code? true
   :conf (nn/builder
          :seed 123
          :optimization-algo :stochastic-gradient-descent
          :iterations 1
          :default-learning-rate 0.006
          :default-updater :nesterovs
          :default-momentum 0.9
          :regularization? true
          :default-l2 1e-4
          :build? true
          :layers {0 {:dense-layer {:n-in 784
                                    :n-out 1000
                                    :updater :nesterovs
                                    :activation-fn :relu
                                    :weight-init :xavier}}
                   1 {:output-layer {:loss-fn :negativeloglikelihood
                                     :n-in 1000
                                     :n-out 10
                                     :updater :nesterovs
                                     :activation-fn :soft-max
                                     :weight-init :xavier}}})))

(def mln
  (new-multi-layer-network
   :conf (nn/builder
          :seed 123
          :optimization-algo :stochastic-gradient-descent
          :iterations 1
          :default-learning-rate 0.006
          :default-updater :nesterovs
          :default-momentum 0.9
          :regularization? true
          :default-l2 1e-4
          :build? true
          :as-code? false
          :layers {0 {:dense-layer {:n-in 784
                                    :n-out 1000
                                    :updater :nesterovs
                                    :activation-fn :relu
                                    :weight-init :xavier}}
                   1 {:output-layer {:loss-fn :negativeloglikelihood
                                     :n-in 1000
                                     :n-out 10
                                     :updater :nesterovs
                                     :activation-fn :soft-max
                                     :weight-init :xavier}}})))

(def init-mln (init! :model mln))

(def fs (new-filesplit :path "resources/poker-hand-training.csv"))

(def rr (initialize-rr! :rr (new-csv-record-reader)
                        :input-split fs))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing return type of termination conditions
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/package-summary.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest termination-condition-creation-test
  (testing "the creation of termination conditions"
    (is (= org.deeplearning4j.earlystopping.termination.InvalidScoreIterationTerminationCondition
           (type (new-invalid-score-iteration-termination-condition))))
    (is (= org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition
           (type (new-max-score-iteration-termination-condition :max-score 10))))
    (is (= org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition
           (type (new-max-time-iteration-termination-condition
                  :max-time-val 10
                  :max-time-unit :seconds))))
    (is (= org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition
           (type (new-score-improvement-epoch-termination-condition
                  :max-n-epoch-no-improve 10
                  :min-improve 5.0))))
    (is (= org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition
           (type (new-score-improvement-epoch-termination-condition
                  :max-n-epoch-no-improve 10))))
    (is (= org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition
           (type (new-best-score-epoch-termination-condition
                  :best-expected-score 2.0))))
    (is (= org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition
           (type (new-best-score-epoch-termination-condition
                  :best-expected-score 2.0
                  :less-is-better? false))))
    (is (= org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
           (type (new-max-epochs-termination-condition :max-n 5))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing return type of epoch termination condition interface
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/EpochTerminationCondition.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest epoch-term-interface-test
  (testing "the return types of the fns found in the epoch termination condition interface"
    (let [max-epochs (new-max-epochs-termination-condition :max-n 5)
          best-score (new-best-score-epoch-termination-condition
                      :best-expected-score 2.0)
          score-imporve (new-score-improvement-epoch-termination-condition
                         :max-n-epoch-no-improve 10)]
      (is (= (type max-epochs) (type (initialize-epoch! max-epochs))))
      (is (= (type best-score) (type (initialize-epoch! best-score))))
      (is (= (type score-imporve) (type (initialize-epoch! score-imporve))))

      (is (= java.lang.Boolean (type
                                (terminate-now-epoch?
                                 :epoch-term-cond max-epochs
                                 :epoch-n 5
                                 :score 3.0))))
      (is (= java.lang.Boolean (type
                                (terminate-now-epoch?
                                 :epoch-term-cond best-score
                                 :epoch-n 5
                                 :score 3.0))))
      (is (= java.lang.Boolean (type
                                (terminate-now-epoch?
                                 :epoch-term-cond score-imporve
                                 :epoch-n 5
                                 :score 3.0)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing return type of iteration termination condition interface
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/IterationTerminationCondition.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest iteration-term-interface-test
  (testing "the return types of the fns found in the iteration termination condition interface"
    (let [invalid-score (new-invalid-score-iteration-termination-condition)
          max-score (new-max-score-iteration-termination-condition :max-score 2.0)
          max-time (new-max-time-iteration-termination-condition :max-time-val 50
                                                                 :max-time-unit :seconds)]
      (is (= (type invalid-score) (type (initialize-iteration! invalid-score))))
      (is (= (type max-score) (type (initialize-iteration! max-score))))
      (is (= (type max-time) (type (initialize-iteration! max-time))))

      (is (= java.lang.Boolean (type
                                (terminate-now-iteration? :iter-term-cond invalid-score
                                                          :last-mini-batch-score 3.0))))
      (is (= java.lang.Boolean (type
                                (terminate-now-iteration? :iter-term-cond max-score
                                                          :last-mini-batch-score 3.0))))
      (is (= java.lang.Boolean (type
                                (terminate-now-iteration? :iter-term-cond max-time
                                                          :last-mini-batch-score 3.0)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of score calculators
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/scorecalc/DataSetLossCalculator.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest score-calc-test
  (testing "the creation of loss calculators"
    (is (= org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
           (type
            (new-ds-loss-calculator
             :iter (new-record-reader-dataset-iterator
                       :record-reader rr
                       :batch-size 5)
             :average? true))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the return type of the score-calc interface
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/scorecalc/ScoreCalculator.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest score-calc-interface-test
  (testing "the return type of the fns found in the score calculator interface"
    ;; this test takes about 15 seconds
    (let [calcer (new-ds-loss-calculator
                :iter (new-mnist-data-set-iterator
                       :batch-size 10
                       :train? true
                       :seed 123)
                :average? true)]
      (is (= java.lang.Double
             (type
              (calculate-score :score-calculator calcer
                               :mln (score!
                                     :model init-mln
                                     :dataset (new-mnist-ds)
                                     :return-model? true))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of model savers
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/saver/package-summary.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest model-savers-test
  (testing "the creation of model savers"
    (is (= org.deeplearning4j.earlystopping.saver.InMemoryModelSaver
           (type (new-in-memory-saver))))
    (is (= org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
           (type (new-local-file-model-saver :directory "resources/temp/testing"))))
    (is (= org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
           (type
            (new-local-file-model-saver :directory "resources/temp/testing"
                                        :charset (Charset/defaultCharset)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the return type of the model savers interface
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/EarlyStoppingModelSaver.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest model-savers-iterface-test
  (testing "the return type of the fns found in the model saver interface"
    (let [empty-saver (new-in-memory-saver)
          non-empty-saver (new-in-memory-saver)]
      ;; we havent saved any models
      (is (= nil (get-best-model empty-saver)))
      (is (= nil (get-latest-model empty-saver)))

      ;; lets save a model
      (is (= (type non-empty-saver) (type
                                     (save-best-model! :saver non-empty-saver
                                                       :net mln
                                                       :score 5.0))))
      (is (= (type non-empty-saver) (type
                                     (save-latest-model! :saver non-empty-saver
                                                         :net mln
                                                         :score 4.0))))
      ;; and show that it got saved
      (is (= org.deeplearning4j.nn.multilayer.MultiLayerNetwork
             (type (get-best-model non-empty-saver))))
      (is (= org.deeplearning4j.nn.multilayer.MultiLayerNetwork
             (type (get-latest-model non-empty-saver)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of early stopping configurations
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/EarlyStoppingConfiguration.Builder.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest early-stopping-conf-test
  (testing "the creation of an early stopping configuration"
    (let [epoch-term (new-max-epochs-termination-condition :max-n 5)
          iteration-term (new-invalid-score-iteration-termination-condition)
          model-saver (new-in-memory-saver)
          score-c (new-ds-loss-calculator
                   :iter (new-record-reader-dataset-iterator
                          :record-reader rr
                          :batch-size 5)
                   :average? true)]
      (is (= org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
             (type
              (new-early-stopping-config :epoch-termination-conditions epoch-term
                                         :iteration-termination-conditions iteration-term
                                         :n-epochs 5
                                         :model-saver model-saver
                                         :save-last-model? false
                                         :score-calculator score-c))))
      (is (= org.deeplearning4j.earlystopping.EarlyStoppingConfiguration$Builder
             (type
              (new-early-stopping-config :epoch-termination-conditions epoch-term
                                         :iteration-termination-conditions iteration-term
                                         :n-epochs 5
                                         :model-saver model-saver
                                         :save-last-model? false
                                         :score-calculator score-c
                                         :build? false)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of early stopping results
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/EarlyStoppingResult.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest early-stopping-result-test
  (testing "the creation of an early stopping result obj"
    (is (= org.deeplearning4j.earlystopping.EarlyStoppingResult
           (type
            (new-early-stopping-result :termination-reason :error
                                       :termination-details "foo"
                                       :score-vs-epoch {2.0 0}
                                       :best-model-epoch 0
                                       :best-model-score 2.0
                                       :total-epochs 1
                                       :best-model mln))))
    (is (= org.deeplearning4j.nn.multilayer.MultiLayerNetwork
           (type
            (get-best-model-from-result
             (new-early-stopping-result :termination-reason :error
                                        :termination-details "foo"
                                        :score-vs-epoch {2.0 0}
                                        :best-model-epoch 0
                                        :best-model-score 2.0
                                        :total-epochs 1
                                        :best-model mln)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of early stopping trainer
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/trainer/EarlyStoppingTrainer.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest early-stopping-trainer-test
  (testing "the creation of an early stopping trainer"
    (is (= org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
           (type
               (new-early-stopping-trainer
                :early-stopping-conf
                (new-early-stopping-config
                 :epoch-termination-conditions
                 (new-max-epochs-termination-condition :max-n 5)
                 :iteration-termination-conditions
                 (new-invalid-score-iteration-termination-condition)
                 :n-epochs 5
                 :model-saver (new-in-memory-saver)
                 :save-last-model? false
                 :score-calculator
                 (new-ds-loss-calculator
                  :iter (new-record-reader-dataset-iterator
                         :record-reader rr
                         :batch-size 5)
                  :average? true))
                :mln mln
                :training-dataset-iterator (new-mnist-data-set-iterator
                                            :batch-size 5
                                            :train? true
                                            :seed 123)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the early stopping trainer interface
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/trainer/IEarlyStoppingTrainer.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest early-stopping-trainer-interface
  (testing "the fit-trainer! fn"
    (let [mnist-train (new-mnist-data-set-iterator :batch-size 25 :train? true
                                                   :seed 123 :n-examples 1024
                                                   :binarize? false :shuffle? true)
          mnist-test (new-mnist-data-set-iterator :batch-size 25 :train? false
                                                  :seed 123 :n-examples 512
                                                  :binarize? false :shuffle? true)
          es-conf (new-early-stopping-config
                   :epoch-termination-conditions (new-max-epochs-termination-condition :max-n 50)
                   :iteration-termination-conditions (new-max-time-iteration-termination-condition
                                                      :max-time-val 1
                                                      :max-time-unit :seconds)
                   :n-epochs 1
                   :model-saver (new-in-memory-saver)
                   :save-last-model? false
                   :score-calculator (new-ds-loss-calculator
                                      :iter mnist-test
                                      :average? true))]
      (is (= org.deeplearning4j.earlystopping.EarlyStoppingResult
             (type (fit-trainer!
                    (new-early-stopping-trainer
                            :early-stopping-conf es-conf
                            :mln mln
                            :iter mnist-train))))))))
