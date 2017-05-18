(ns dl4clj.earlystopping-tests
  (:require [dl4clj.earlystopping.early-stopping-config :refer :all]
            [dl4clj.earlystopping.listener :refer :all]))











;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing return type of EarlyStoppingListeners interface
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/listener/EarlyStoppingListener.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest listener-interface-test
  (testing "the fns found in the EarlyStoppingListener interface"
    #_(is (= "" (on-completion! )))
    #_(is (= "" (on-epoch! )))
    #_(is (= "" (on-start! )))
))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing return type of EpochTerminationCondition interface
;; https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/termination/EpochTerminationCondition.html
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest epoch-termination-condition-test
  (testing "the fns found in th eEpochTerminationCondition interface"



    ))
