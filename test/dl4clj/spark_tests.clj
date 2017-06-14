(ns dl4clj.spark-tests
  (:refer-clojure :exclude [rand])
  (:require [clojure.test :refer :all]
            [dl4clj.spark.data.data-fns :refer :all]
            [dl4clj.spark.datavec.datavec-fns :refer :all]

            ;; other fns used for testing
            [dl4clj.datasets.iterator.impl.default-datasets :refer [new-iris-data-set-iterator]]
            ;; reseting the iter
            [nd4clj.linalg.dataset.api.iterator.data-set-iterator :refer [reset-iter!
                                                                          has-next?]]
            ;; multi-ds
            [nd4clj.linalg.dataset.multi-ds :refer [new-multi-ds]]

            ;; data for multi-ds
            [nd4clj.linalg.factory.nd4j :refer [rand]]

            ;; used in multi-ds-iter
            [dl4clj.utils :refer [array-of]]

            ;; need record readers
            [datavec.api.records.readers :refer [new-csv-record-reader
                                                 new-csv-nlines-seq-record-reader]]

            ;; ds pre-processor
            [nd4clj.linalg.dataset.api.pre-processors :refer [new-min-max-normalization-ds-preprocessor]]

            ;; string split
            [datavec.api.split :refer [new-string-split
                                       get-list-string-split-data]]
            )
  (:import [org.nd4j.linalg.dataset.api.iterator TestMultiDataSetIterator]
           [org.nd4j.linalg.dataset.api MultiDataSet]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; helper fns and defs used in testing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def iris-iter (new-iris-data-set-iterator :batch-size 10 :n-examples 2))
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

(defn consistent-ds-save
  [iter]
  (-> iter
      reset-iter?!
      .next
      (.save (clojure.java.io/as-file "resources/tests/spark/test-save.bin"))))

(defn consistent-multi-ds-save
  [multi-iter]
  (-> multi-iter
      reset-multi-ds-iter?!
      .next
      (.save (clojure.java.io/as-file "resources/tests/spark/test-save-multi.bin"))))

(def csv-rr (new-csv-record-reader))

(def poker-training-file-byte-size
  (int (.length (clojure.java.io/as-file "resources/poker-spark-test.csv"))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; data-fns creation and calling
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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
    (is (= '(:export-fn :iter)
           (keys (call-ds-export-fn! :the-fn (new-ds-export-fn :export-path "resources/tests/spark/only-export/")
                                     :ds-iter (reset-iter?! iris-iter)))))
    (is (= '(:export-fn :iter)
           (keys (call-ds-export-fn! :the-fn {:export-ds {:export-path "resources/tests/spark/only-export/"}}
                                     :ds-iter (reset-iter?! iris-iter)))))
    (is (= '(:export-fn :iter)
           (keys (call-ds-export-fn! :the-fn (new-multi-ds-export-fn :export-path "resources/tests/spark/only-export/")
                                     :ds-iter (reset-multi-ds-iter?! multi-ds-iter)))))
    (is (= '(:export-fn :iter)
           (keys (call-ds-export-fn! :the-fn {:export-multi-ds {:export-path "resources/tests/spark/only-export/"}}
                                     :ds-iter (reset-multi-ds-iter?! multi-ds-iter)))))
    (is (= org.nd4j.linalg.dataset.DataSet
           (type (do (consistent-ds-save iris-iter)
                     (call-path-to-ds-fn! :the-fn (new-path-to-ds-fn)
                                          :path "resources/tests/spark/test-save.bin")))))
    (is (= org.nd4j.linalg.dataset.DataSet
           (type (do (consistent-ds-save iris-iter)
                     (call-path-to-ds-fn! :the-fn {:path-to-ds {}}
                                          :path "resources/tests/spark/test-save.bin")))))

    (is (= org.nd4j.linalg.dataset.MultiDataSet
           (type (do (consistent-multi-ds-save multi-ds-iter)
                     (call-path-to-multi-ds-fn! :the-fn (new-path-to-multi-ds-fn)
                                                :path "resources/tests/spark/test-save-multi.bin")))))
    (is (= org.nd4j.linalg.dataset.MultiDataSet
           (type (do (consistent-multi-ds-save multi-ds-iter)
                     (call-path-to-multi-ds-fn! :the-fn {:path-to-multi-ds {}}
                                                :path "resources/tests/spark/test-save-multi.bin")))))
    (is (= java.util.ArrayList$Itr
           (type (call-split-ds-fn! :the-fn (new-split-ds-fn)
                                    :ds-iter (reset-iter?! iris-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-split-ds-fn! :the-fn {:split-ds {}}
                                    :ds-iter (reset-iter?! iris-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-split-ds-with-appended-key! :the-fn (new-split-ds-with-appended-key :max-key-idx 12)
                                                   :ds (.next (reset-iter?! iris-iter))))))
    (is (= java.util.ArrayList$Itr
           (type (call-split-ds-with-appended-key! :the-fn {:split-ds-rand {:max-key-idx 12}}
                                                   :ds (.next (reset-iter?! iris-iter))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; datavec-fns creation and calling
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(deftest create-datavec-spark-fns
  (testing "the creation of datavec fns for spark"
    (is (= org.deeplearning4j.spark.datavec.RecordReaderFunction
           (type (new-record-reader-fn :record-reader csv-rr
                                       :label-idx 10
                                       :n-labels 10))))
    (is (= org.deeplearning4j.spark.datavec.DataVecByteDataSetFunction
           (type (new-datavec-byte-ds-fn :label-idx 10 :n-labels 10
                                         :batch-size 10 :byte-file-len poker-training-file-byte-size
                                         :regression? false))))
    (is (= org.deeplearning4j.spark.datavec.DataVecDataSetFunction
           (type (new-datavec-ds-fn :label-idx 10 :n-labels 10 :regression? false))))
    (is (= org.deeplearning4j.spark.datavec.DataVecSequenceDataSetFunction
           (type (new-datavec-seq-ds-fn :label-idx 10 :n-labels 10
                                        :regression? false
                                        :pre-processor (new-min-max-normalization-ds-preprocessor)))))
    (is (= org.deeplearning4j.spark.datavec.DataVecSequencePairDataSetFunction
           (type (new-datavec-seq-pair-ds-fn :n-labels 10 :regression? false
                                             :spark-alignment-mode :equal-length))))))

(deftest calling-datavec-spark-fns
  (testing "the calling of the datavec spark fns"
    (is (= org.nd4j.linalg.dataset.DataSet
           (type (call-record-reader-fn! :the-fn (new-record-reader-fn :record-reader csv-rr
                                                                       :label-idx 10
                                                                       :n-labels 10)
                                         :string-ds (slurp "resources/poker-spark-test.csv")))))
    (is (= org.nd4j.linalg.dataset.DataSet
           (type (call-record-reader-fn! :the-fn {:record-reader-fn {:record-reader csv-rr
                                                                     :label-idx 10
                                                                     :n-labels 10}}
                                         :string-ds (slurp "resources/poker-spark-test.csv")))))

    ;; to test fns which require tuple inputs, datavec.spark packages need to be implemented
    #_(is (= "" (call-datavec-byte-ds-fn! :the-fn (new-datavec-byte-ds-fn :label-idx 10
                                                                        :n-labels 10
                                                                        :batch-size 1
                                                                        :byte-file-len poker-training-file-byte-size))))

    ;; implement the higher level implementation fns
    ;; see when we get back to the low level

    ))
