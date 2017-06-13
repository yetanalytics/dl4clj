(ns dl4clj.spark-tests
  (:require [clojure.test :refer :all]
            [dl4clj.spark.data.data-fns :refer :all]

            ;; other fns used for testing
            [dl4clj.datasets.iterator.impl.default-datasets :refer [new-iris-data-set-iterator]]
            ;; reseting the iter
            [nd4clj.linalg.dataset.api.iterator.data-set-iterator :refer [reset-iter!
                                                                          has-next?]]
            ;; multi-ds
            [nd4clj.linalg.dataset.multi-ds :refer [new-multi-ds]]

            ;; data for multi-ds
            ;; dont forget to add in the refer-clojure exclude
            [nd4clj.linalg.factory.nd4j :refer [rand]]
            [dl4clj.utils :refer [array-of]])
  (:import [org.nd4j.linalg.dataset.api.iterator TestMultiDataSetIterator]
           [org.nd4j.linalg.dataset.api MultiDataSet]))

(def iris-iter (new-iris-data-set-iterator :batch-size 10 :n-examples 5))
;; this way of making an iter from an existing multi-dataset needs to
;; be migrated over to the multi-dataset ns or have its own
(def multi-ds-iter (TestMultiDataSetIterator.
                    2
                    (array-of :data (new-multi-ds :features (rand [2 4]) :labels (rand [2 2]))
                              :java-type MultiDataSet)))

(defn reset-iter?!
  [iter]
  (if (false? (has-next? iter))
    (reset-iter! iter)
    iter))

(defn reset-multi-ds-iter?!
  "this will eventually make its way into the multi-ds-iter interface"
  [m-ds-iter]
  (if (.hasNext m-ds-iter)
    m-ds-iter
    (doto m-ds-iter .reset)))



(deftest data-fns-test
  (testing "the creation of ds-fns"
    (is (= org.deeplearning4j.spark.data.BatchAndExportDataSetsFunction
           (type (new-batch-and-export-ds-fn :batch-size 2 :export-path "resources/tmp/"))))
    (is (= org.deeplearning4j.spark.data.BatchAndExportMultiDataSetsFunction
           (type (new-batch-and-export-multi-ds-fn :batch-size 2 :export-path "resources/tmp/"))))
    (is (= org.deeplearning4j.spark.data.BatchDataSetsFunction
           (type (new-batch-ds-fn :batch-size 2))))
    (is (= org.deeplearning4j.spark.data.DataSetExportFunction
           (type (new-ds-export-fn :export-path "resources/tmp/"))))
    (is (= org.deeplearning4j.spark.data.MultiDataSetExportFunction
           (type (new-multi-ds-export-fn :export-path "resources/tmp/"))))
    (is (= org.deeplearning4j.spark.data.PathToDataSetFunction
           (type (new-path-to-ds-fn))))
    (is (= org.deeplearning4j.spark.data.PathToMultiDataSetFunction
           (type (new-path-to-multi-ds-fn))))
    (is (= org.deeplearning4j.spark.data.SplitDataSetsFunction
           (type (new-split-ds-fn))))
    (is (= org.deeplearning4j.spark.data.shuffle.SplitDataSetExamplesPairFlatMapFunction
           (type (new-split-ds-with-appended-key :max-key-idx 4))))))

(deftest calling-data-fns-test
  (testing "using the call-ds-fns! fn with a ds iter"
    (is (= java.util.ArrayList$Itr
           (type (call-batch-and-export-ds-fn!
                  :the-fn (new-batch-and-export-ds-fn
                           :batch-size 1
                           :export-path "resources/tests/spark/")
                  :partition-idx 1
                  :ds-iter (reset-iter?! iris-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-batch-and-export-ds-fn!
                  :the-fn {:batch-and-export-ds
                           {:batch-size 1
                            :export-path "resources/tests/spark/"}}
                  :partition-idx 2
                  :ds-iter (reset-iter?! iris-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-batch-and-export-multi-ds-fn!
                  :the-fn (new-batch-and-export-multi-ds-fn
                           :batch-size 1
                           :export-path "resources/tests/spark/")
                  :partition-idx 3
                  :multi-ds-iter (reset-multi-ds-iter?! multi-ds-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-batch-and-export-multi-ds-fn!
                  :the-fn {:batch-and-export-multi-ds
                           {:batch-size 1
                            :export-path "resources/tests/spark/"}}
                  :partition-idx 4
                  :multi-ds-iter (reset-multi-ds-iter?! multi-ds-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-batch-ds-fn! :the-fn (new-batch-ds-fn :batch-size 1)
                                    :ds-iter (reset-iter?! iris-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-batch-ds-fn! :the-fn {:batch-ds {:batch-size 1}}
                                    :ds-iter (reset-iter?! iris-iter)))))
    ))
