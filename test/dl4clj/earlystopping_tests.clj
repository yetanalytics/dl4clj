(ns dl4clj.earlystopping-tests
  (:require [dl4clj.earlystopping.early-stopping-config :refer [new-early-stopping-config]]
            [dl4clj.earlystopping.early-stopping-trainer :refer [new-early-stopping-trainer]]
            [dl4clj.earlystopping.score-calc :refer [new-ds-loss-calculator]]
            [dl4clj.earlystopping.termination-conditions :refer :all]
            [dl4clj.earlystopping.model-saver :refer [new-in-memory-saver new-local-file-model-saver]]
            [dl4clj.earlystopping.api.early-stopping-trainer :refer :all]
            [dl4clj.earlystopping.api.model-saver :refer :all]
            [dl4clj.earlystopping.api.score-calc :refer :all]
            ;; namespaces I need to test the above namespaces
            [dl4clj.nn.multilayer.multi-layer-network :refer [new-multi-layer-network]]
            [dl4clj.nn.conf.builders.nn :as nn]
            [dl4clj.nn.api.model :refer [init! score!]]
            [dl4clj.datasets.iterators :refer [new-record-reader-dataset-iterator
                                               new-mnist-data-set-iterator]]
            [dl4clj.datasets.record-readers :refer [new-csv-record-reader]]
            [dl4clj.datasets.input-splits :refer [new-filesplit]]
            [dl4clj.datasets.record-readers :refer [new-csv-record-reader]]
            [dl4clj.datasets.api.record-readers :refer [initialize-rr!]]
            [dl4clj.datasets.default-datasets :refer [new-mnist-ds]]
            [dl4clj.utils :refer [as-code]]
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
                                     :weight-init :xavier}}})
   :as-code? false))

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
           (type (new-invalid-score-iteration-termination-condition :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.termination.InvalidScoreIterationTerminationCondition.)
           (new-invalid-score-iteration-termination-condition)))

    (is (= org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition
           (type (new-max-score-iteration-termination-condition :max-score 10 :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition. 10)
           (new-max-score-iteration-termination-condition :max-score 10)))

    (is (= org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition
           (type (new-max-time-iteration-termination-condition
                  :max-time-val 10
                  :max-time-unit :seconds
                  :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition.
             10 (dl4clj.constants/value-of {:time-unit :seconds}))
           (new-max-time-iteration-termination-condition
            :max-time-val 10
            :max-time-unit :seconds)))

    (is (= org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition
           (type (new-score-improvement-epoch-termination-condition
                  :max-n-epoch-no-improve 10
                  :min-improve 5.0
                  :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition. 10 5.0)
           (new-score-improvement-epoch-termination-condition
            :max-n-epoch-no-improve 10
            :min-improve 5.0)))

    (is (= org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition
           (type (new-score-improvement-epoch-termination-condition
                  :max-n-epoch-no-improve 10
                  :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition. 10)
           (new-score-improvement-epoch-termination-condition
            :max-n-epoch-no-improve 10)))

    (is (= org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition
           (type (new-best-score-epoch-termination-condition
                  :best-expected-score 2.0
                  :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition. 2.0)
           (new-best-score-epoch-termination-condition
            :best-expected-score 2.0)))

    (is (= org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition
           (type (new-best-score-epoch-termination-condition
                  :best-expected-score 2.0
                  :is-less-better? false
                  :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.termination.BestScoreEpochTerminationCondition. 2.0 false)
           (new-best-score-epoch-termination-condition
            :best-expected-score 2.0
            :is-less-better? false)))

    (is (= org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition
           (type (new-max-epochs-termination-condition :max-n 5 :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition. 5)
           (new-max-epochs-termination-condition :max-n 5)))))

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
             :average? true
             :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator.
             (org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator.
              (clojure.core/doto
                  (org.datavec.api.records.reader.impl.csv.CSVRecordReader.)
                (.initialize
                 (org.datavec.api.split.FileSplit.
                  (clojure.java.io/as-file "resources/poker-hand-training.csv"))))
              5)
             true)
           (new-ds-loss-calculator
            :iter (new-record-reader-dataset-iterator
                   :record-reader rr
                   :batch-size 5)
            :average? true)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the return type of the score-calc interface
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/scorecalc/ScoreCalculator.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest score-calc-interface-test
  (testing "the return type of the fns found in the score calculator interface"
    ;; this test takes about 15 seconds
    (let [calcer (new-ds-loss-calculator
                  :as-code? false
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
                                     :dataset (new-mnist-ds :as-code? false)
                                     :return-model? true))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing the creation of model savers
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/saver/package-summary.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest model-savers-test
  (testing "the creation of model savers"
    (is (= org.deeplearning4j.earlystopping.saver.InMemoryModelSaver
           (type (new-in-memory-saver :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.saver.InMemoryModelSaver.)
           (new-in-memory-saver)))

    (is (= org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
           (type (new-local-file-model-saver :directory "resources/temp/testing"
                                             :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.saver.LocalFileModelSaver. "resources/temp/testing")
           (new-local-file-model-saver :directory "resources/temp/testing")))

    (is (= org.deeplearning4j.earlystopping.saver.LocalFileModelSaver
           (type
            (new-local-file-model-saver :directory "resources/temp/testing"
                                        :charset `(Charset/defaultCharset)
                                        :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.saver.LocalFileModelSaver.
             "resources/temp/testing" (java.nio.charset.Charset/defaultCharset))
           (new-local-file-model-saver :directory "resources/temp/testing"
                                       :charset `(Charset/defaultCharset))))))

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
                                         :eval-every-n-epochs 5
                                         :model-saver model-saver
                                         :save-last-model? false
                                         :score-calculator score-c
                                         :as-code? false))))
      (is (= '(.build
               (doto
                   (org.deeplearning4j.earlystopping.EarlyStoppingConfiguration$Builder.)
                 (.evaluateEveryNEpochs 5)
                 (.scoreCalculator
                  (org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator.
                   (org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator.
                    (clojure.core/doto
                        (org.datavec.api.records.reader.impl.csv.CSVRecordReader.)
                      (.initialize
                       (org.datavec.api.split.FileSplit.
                        (clojure.java.io/as-file
                         "resources/poker-hand-training.csv"))))
                    5)
                   true))
                 (.saveLastModel false)
                 (.modelSaver
                  (clojure.core/first
                   (dl4clj.utils/array-of
                    :java-type
                    org.deeplearning4j.earlystopping.EarlyStoppingModelSaver
                    :data
                    (org.deeplearning4j.earlystopping.saver.InMemoryModelSaver.))))
                 (.iterationTerminationConditions
                  (dl4clj.utils/array-of
                   :data
                   (org.deeplearning4j.earlystopping.termination.InvalidScoreIterationTerminationCondition.)
                   :java-type
                   org.deeplearning4j.earlystopping.termination.IterationTerminationCondition))
                 (.epochTerminationConditions
                  (dl4clj.utils/array-of
                   :data
                   (org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition.
                    5)
                   :java-type
                   org.deeplearning4j.earlystopping.termination.EpochTerminationCondition))))
             (new-early-stopping-config :epoch-termination-conditions epoch-term
                                        :iteration-termination-conditions iteration-term
                                        :eval-every-n-epochs 5
                                        :model-saver model-saver
                                        :save-last-model? false
                                        :score-calculator score-c))))))

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
              :eval-every-n-epochs 5
              :model-saver (new-in-memory-saver)
              :save-last-model? false
              :score-calculator
              (new-ds-loss-calculator
               :iter (new-record-reader-dataset-iterator
                      :record-reader rr
                      :batch-size 5)
               :average? true))
             :mln mln-code
             :iter (new-mnist-data-set-iterator
                    :batch-size 5
                    :train? true
                    :seed 123)
             :as-code? false))))
    (is (= '(org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer.
             (.build
              (doto
                  (org.deeplearning4j.earlystopping.EarlyStoppingConfiguration$Builder.)
                (.evaluateEveryNEpochs 5)
                (.scoreCalculator
                 (org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator.
                  (org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator.
                   (clojure.core/doto
                       (org.datavec.api.records.reader.impl.csv.CSVRecordReader.)
                     (.initialize
                      (org.datavec.api.split.FileSplit.
                       (clojure.java.io/as-file "resources/poker-hand-training.csv"))))
                   5)
                  true))
                (.saveLastModel false)
                (.modelSaver
                 (clojure.core/first
                  (dl4clj.utils/array-of
                   :java-type
                   org.deeplearning4j.earlystopping.EarlyStoppingModelSaver
                   :data
                   (org.deeplearning4j.earlystopping.saver.InMemoryModelSaver.))))
                (.iterationTerminationConditions
                 (dl4clj.utils/array-of
                  :data
                  (org.deeplearning4j.earlystopping.termination.InvalidScoreIterationTerminationCondition.
                   )
                  :java-type
                  org.deeplearning4j.earlystopping.termination.IterationTerminationCondition))
                (.epochTerminationConditions
                 (dl4clj.utils/array-of
                  :data
                  (org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition.
                   5)
                  :java-type
                  org.deeplearning4j.earlystopping.termination.EpochTerminationCondition))))
             (org.deeplearning4j.nn.multilayer.MultiLayerNetwork.
              (.build
               (doto
                   (.list
                    (doto
                        (org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder.)
                      (.l2 1.0E-4)
                      (.regularization true)
                      (.updater (dl4clj.constants/value-of {:updater :nesterovs}))
                      (.seed 123)
                      (.momentum 0.9)
                      (.iterations 1)
                      (.learningRate 0.006)
                      (.optimizationAlgo
                       (dl4clj.constants/value-of
                        {:optimization-algorithm :stochastic-gradient-descent}))))
                 (.layer
                  0
                  (dl4clj.utils/eval-and-build
                   (dl4clj.nn.conf.builders.layers/builder
                    {:dense-layer
                     {:n-in 784,
                      :n-out 1000,
                      :updater
                      :nesterovs,
                      :activation-fn :relu,
                      :weight-init :xavier}})))
                 (.layer
                  1
                  (dl4clj.utils/eval-and-build
                   (dl4clj.nn.conf.builders.layers/builder
                    {:output-layer
                     {:loss-fn :negativeloglikelihood,
                      :n-in 1000,
                      :n-out 10,
                      :updater :nesterovs,
                      :activation-fn :soft-max,
                      :weight-init :xavier}}))))))
             (org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator.
              5
              true
              123))
           (new-early-stopping-trainer
            :as-code? true
            :early-stopping-conf
            (new-early-stopping-config
             :epoch-termination-conditions
             (new-max-epochs-termination-condition :max-n 5)
             :iteration-termination-conditions
             (new-invalid-score-iteration-termination-condition)
             :eval-every-n-epochs 5
             :model-saver (new-in-memory-saver)
             :save-last-model? false
             :score-calculator
             (new-ds-loss-calculator
              :iter (new-record-reader-dataset-iterator
                     :record-reader rr
                     :batch-size 5)
              :average? true))
            :mln mln-code
            :iter (new-mnist-data-set-iterator
                   :batch-size 5
                   :train? true
                   :seed 123))))))

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
                   :eval-every-n-epochs 1
                   :model-saver (new-in-memory-saver)
                   :save-last-model? false
                   :score-calculator (new-ds-loss-calculator
                                      :iter mnist-test
                                      :average? true))]
      (is (= org.deeplearning4j.earlystopping.EarlyStoppingResult
             (type (fit-trainer!
                    (new-early-stopping-trainer
                            :early-stopping-conf es-conf
                            :mln mln-code
                            :iter mnist-train
                            :as-code? false))))))))
