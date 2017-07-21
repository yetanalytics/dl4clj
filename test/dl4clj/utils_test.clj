(ns dl4clj.utils-test
  (:require [clojure.test :refer :all]
            [dl4clj.utils :refer :all]
            [dl4clj.constants :refer :all]))

(deftest util-tests
  (testing "the fns in utils"
    (is (= :fooBar (camelize :foo-bar )))
    (is (= :FooBar (camelize :foo-bar true)))
    (is (= "fooBar" (camelize "foo-bar")))
    (is (= "fooBar" (camelize "foo bar")))
    (is (= "FooBar" (camelize "Foo Bar")))
    (is (= "foo-bar" (camel-to-dashed "fooBar")))
    (is (= '([1 0] [2 1] [3 2] [4 3]) (indexed [1 2 3 4])))
    ))

(deftest builder-fn-tests
  (testing "the build fn"
    (is (= '(doto "foo"
              (.aMethodBooleanArg true)
              (.aMethodNumberArg 0.2)
              (.aMethodRequiresOutputOfFnTakesConfigMap (a-fn-to-call {:some-config {:more-config "opts"}}))
              (.aMethodRequiresOutputOfFnTakesKeyword (a-fn-to-call :some-keyword))
              (.aMethodWhichNeedsManyArgs "first val" "second val" "third val")
              (.aMethodWhichNeedsMap {0 0.2, 1 0.4})
              (.aMethodWhichNeedsNestedMap {:foo {0 0.2, 1 0.4}}))
           (builder-fn
            "foo"
            {:boolean-arg             '.aMethodBooleanArg
             :number-arg              '.aMethodNumberArg
             :fn-needs-map            '.aMethodRequiresOutputOfFnTakesConfigMap
             :fn-needs-keyword        '.aMethodRequiresOutputOfFnTakesKeyword
             :multi-arg               '.aMethodWhichNeedsManyArgs
             :method-needs-map        '.aMethodWhichNeedsMap
             :method-needs-nested-map '.aMethodWhichNeedsNestedMap}

            {:boolean-arg             true
             :number-arg              0.2
             :fn-needs-map            '(a-fn-to-call {:some-config {:more-config "opts"}})
             :fn-needs-keyword        '(a-fn-to-call :some-keyword)
             :multi-arg               ["first val" "second val" "third val"]
             :method-needs-map        {0 0.2 1 0.4}
             :method-needs-nested-map {:foo {0 0.2 1 0.4}}})))
    (is (= {:foo "zab" :bar "bell"} (replace-map-vals {:foo "baz" :bar "bell"} {:foo "zab"})))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; constants, value-of, input-type
;; dl4clj.constants
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest enum-testing
  (testing "the creation of dl4j enums/constants"
    ;; constants (helper fn)
    (is (= "RELU" (constants #(str %) :relu :activation? true)))
    (is (= "FooBaz" (constants #(str %) :foo-baz :camel? true)))
    (is (= "Relu" (constants #(str %) :relu :activation? true :camel? true)))
    (is (= "FOO_BAZ" (constants #(str %) :foo-baz)))

    (is (= org.nd4j.linalg.activations.Activation
           (type (value-of {:activation-fn :relu}))))
    (is (= org.deeplearning4j.nn.conf.GradientNormalization
           (type (value-of {:gradient-normalization :none}))))
    (is (= org.deeplearning4j.nn.conf.LearningRatePolicy
           (type (value-of {:learning-rate-policy :poly}))))
    (is (= org.deeplearning4j.nn.conf.Updater
           (type (value-of {:updater :adam}))))
    (is (= org.deeplearning4j.nn.weights.WeightInit
           (type (value-of {:weight-init :xavier}))))
    (is (= org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction
           (type (value-of {:loss-fn :mse}))))
    (is (= org.deeplearning4j.nn.conf.layers.RBM$HiddenUnit
           (type (value-of {:hidden-unit :binary}))))
    (is (= org.deeplearning4j.nn.conf.layers.RBM$VisibleUnit
           (type (value-of {:visible-unit :binary}))))
    (is (= org.deeplearning4j.nn.conf.ConvolutionMode
           (type (value-of {:convolution-mode :strict}))))
    (is (= org.deeplearning4j.nn.conf.layers.ConvolutionLayer$AlgoMode
           (type (value-of {:cudnn-algo-mode :no-workspace}))))
    (is (= org.deeplearning4j.nn.conf.layers.PoolingType
           (type (value-of {:pool-type :avg}))))
    (is (= org.deeplearning4j.nn.conf.BackpropType
           (type (value-of {:backprop-type :standard}))))
    (is (= org.deeplearning4j.nn.api.OptimizationAlgorithm
           (type (value-of {:optimization-algorithm :lbfgs}))))
    (is (= org.deeplearning4j.nn.api.MaskState
           (type (value-of {:mask-state :active}))))
    (is (= org.deeplearning4j.nn.api.Layer$Type
           (type (value-of {:layer-type :feed-forward}))))
    (is (= org.deeplearning4j.nn.api.Layer$TrainingMode
           (type (value-of {:layer-training-mode :train}))))
    (is (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeRecurrent
           (type (input-types {:recurrent {:size 10}}))))
    (is (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeFeedForward
           (type (input-types {:feed-forward {:size 10}}))))
    (is (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeConvolutional
           (type (input-types {:convolutional {:height 1 :width 1 :depth 1}}))))
    (is (= org.deeplearning4j.nn.conf.inputs.InputType$InputTypeConvolutionalFlat
           (type (input-types {:convolutional-flat {:height 1 :width 1 :depth 1}}))))
    (is (= org.deeplearning4j.spark.api.RDDTrainingApproach
           (type (value-of {:rdd-training-approach :direct}))))
    (is (= org.deeplearning4j.spark.api.RDDTrainingApproach
           (type (value-of {:rdd-training-approach :export}))))
    (is (= org.deeplearning4j.spark.api.Repartition
           (type (value-of {:repartition :always}))))
    (is (= org.deeplearning4j.spark.api.Repartition
           (type (value-of {:repartition :never}))))
    (is (= org.deeplearning4j.spark.api.Repartition
           (type (value-of {:repartition :num-partitions-workers-differs}))))
    (is (= org.deeplearning4j.spark.api.RepartitionStrategy
           (type (value-of {:repartition-strategy :balanced}))))
    (is (= org.deeplearning4j.spark.api.RepartitionStrategy
           (type (value-of {:repartition-strategy :spark-default}))))
    (is (= org.deeplearning4j.spark.datavec.DataVecSequencePairDataSetFunction$AlignmentMode
           (type (value-of {:spark-alignment-mode :align-end}))))
    (is (= org.deeplearning4j.spark.datavec.DataVecSequencePairDataSetFunction$AlignmentMode
           (type (value-of {:spark-alignment-mode :align-start}))))
    (is (= org.deeplearning4j.spark.datavec.DataVecSequencePairDataSetFunction$AlignmentMode
           (type (value-of {:spark-alignment-mode :equal-length}))))


    ))
