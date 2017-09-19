(ns dl4clj.spark-tests
  (:require [clojure.test :refer :all]
            [dl4clj.spark.data.data-fns :refer :all]
            [dl4clj.spark.datavec.datavec-fns :refer :all]

            ;; file split for data

            [dl4clj.datasets.input-splits :refer [new-filesplit]]

            ;; record reader init

            [dl4clj.datasets.api.record-readers :refer [initialize-rr!]]

            ;; other fns used for testing
            [dl4clj.datasets.iterators :refer [new-iris-data-set-iterator
                                               new-existing-dataset-iterator]]
            ;; reseting the iter
            [dl4clj.datasets.api.iterators :refer [reset-iter!
                                                   has-next?
                                                   next-example!]]
            [dl4clj.helpers :refer [data-from-iter
                                    reset-iterator!]]
            ;; multi-ds
            [dl4clj.datasets.new-datasets :refer [new-multi-ds new-ds]]
            [dl4clj.datasets.api.datasets :refer [new-ds-iter]]

            ;; data for multi-ds
            [nd4clj.linalg.factory.nd4j :refer [indarray-of-rand]]

            ;; used in multi-ds-iter
            [dl4clj.utils :refer [array-of as-code]]

            ;; need record readers
            [dl4clj.datasets.record-readers :refer [new-csv-record-reader
                                                    new-csv-nlines-seq-record-reader]]

            ;; ds pre-processor
            [dl4clj.datasets.pre-processors :refer [new-min-max-normalization-ds-preprocessor]]

            ;; string split
            [dl4clj.datasets.input-splits :refer [new-string-split]]

            [dl4clj.datasets.api.input-splits :refer [get-list-string-split-data]]

            [dl4clj.spark.data.dataset-provider :refer [spark-context-to-dataset-rdd]])
  (:import [org.nd4j.linalg.dataset.api.iterator TestMultiDataSetIterator]
           [org.nd4j.linalg.dataset.api MultiDataSet]
           [org.apache.spark SparkConf]
           [org.apache.spark.api.java JavaRDD JavaSparkContext]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder]))

;; remove things that arn't going to be apart of core
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; helper fns and defs used in testing
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def hadoop-home-dir (System/setProperty "hadoop.home.dir" "/"))

(def iris-iter (new-iris-data-set-iterator :batch-size 1 :n-examples 5 :as-code? false))

;; this way of making an iter from an existing multi-dataset needs to
;; be migrated over to the multi-dataset ns or have its own

(def multi-ds-iter
  (TestMultiDataSetIterator.
   2
   (array-of :data (new-multi-ds
                    :features (as-code indarray-of-rand :rows 2 :columsn 4)
                    :labels (as-code indarray-of-rand :rows 2 :columns 2)
                    :as-code? false)
             :java-type MultiDataSet)))

(defn consistent-ds-save
  [iter]
  (-> iter
      reset-iterator!
      .next
      (.save (clojure.java.io/as-file "resources/tests/spark/test-save.bin"))))

(defn consistent-multi-ds-save
  [multi-iter]
  (-> multi-iter
      reset-iterator!
      .next
      (.save (clojure.java.io/as-file "resources/tests/spark/test-save-multi.bin"))))

(def fs (new-filesplit :path "resources/poker-hand-training.csv"))

(def csv-rr (initialize-rr! :rr (new-csv-record-reader)
                            :input-split fs
                            :as-code? false))

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
                  :iter (reset-iterator! iris-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-batch-and-export-ds-fn!
                  :the-fn {:batch-and-export-ds
                           {:batch-size 1
                            :export-path "resources/tests/spark/"}}
                  :partition-idx 2
                  :iter (reset-iterator! iris-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-batch-and-export-multi-ds-fn!
                  :the-fn (new-batch-and-export-multi-ds-fn
                           :batch-size 1
                           :export-path "resources/tests/spark/")
                  :partition-idx 3
                  :multi-ds-iter (reset-iterator! multi-ds-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-batch-and-export-multi-ds-fn!
                  :the-fn {:batch-and-export-multi-ds
                           {:batch-size 1
                            :export-path "resources/tests/spark/"}}
                  :partition-idx 4
                  :multi-ds-iter (reset-iterator! multi-ds-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-batch-ds-fn! :the-fn (new-batch-ds-fn :batch-size 1)
                                    :iter (reset-iterator! iris-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-batch-ds-fn! :the-fn {:batch-ds {:batch-size 1}}
                                    :iter (reset-iterator! iris-iter)))))
    (is (= '(:export-fn :iter)
           (keys (call-ds-export-fn! :the-fn (new-ds-export-fn :export-path "resources/tests/spark/only-export/")
                                     :iter (reset-iterator! iris-iter)))))
    (is (= '(:export-fn :iter)
           (keys (call-ds-export-fn! :the-fn {:export-ds {:export-path "resources/tests/spark/only-export/"}}
                                     :iter (reset-iterator! iris-iter)))))
    (is (= '(:export-fn :iter)
           (keys (call-ds-export-fn! :the-fn (new-multi-ds-export-fn :export-path "resources/tests/spark/only-export/")
                                     :iter (reset-iterator! multi-ds-iter)))))
    (is (= '(:export-fn :iter)
           (keys (call-ds-export-fn! :the-fn {:export-multi-ds {:export-path "resources/tests/spark/only-export/"}}
                                     :iter (reset-iterator! multi-ds-iter)))))
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
                                    :iter (reset-iterator! iris-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-split-ds-fn! :the-fn {:split-ds {}}
                                    :iter (reset-iterator! iris-iter)))))
    (is (= java.util.ArrayList$Itr
           (type (call-split-ds-with-appended-key! :the-fn (new-split-ds-with-appended-key :max-key-idx 12)
                                                   :ds (.next (reset-iterator! iris-iter))))))
    (is (= java.util.ArrayList$Itr
           (type (call-split-ds-with-appended-key! :the-fn {:split-ds-rand {:max-key-idx 12}}
                                                   :ds (.next (reset-iterator! iris-iter))))))))

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
                                             :spark-alignment-mode :equal-length))))
    (is (= org.deeplearning4j.spark.datavec.export.StringToDataSetExportFunction
           (type (new-string-to-ds-export-fn :output-directory "resources/tests/spark/export/"
                                             :record-reader csv-rr
                                             :batch-size 1
                                             :regression? false
                                             :label-idx 10
                                             :n-labels 10))))))

(deftest calling-datavec-spark-fns
  (testing "the calling of the datavec spark fns"
    (is (= org.nd4j.linalg.dataset.DataSet
           (type (call-record-reader-fn!
                  :the-fn
                  (new-record-reader-fn :record-reader csv-rr
                                        :label-idx 10
                                        :n-labels 10)
                  :string-ds (slurp "resources/poker-spark-test.csv")))))
    (is (= org.nd4j.linalg.dataset.DataSet
           (type (call-record-reader-fn! :the-fn {:record-reader-fn {:record-reader csv-rr
                                                                     :label-idx 10
                                                                     :n-labels 10}}
                                         :string-ds (slurp "resources/poker-spark-test.csv")))))
    (is (= '(:fn :iter)
           (keys (call-string-to-ds-export-fn!
                  :the-fn (new-string-to-ds-export-fn
                           :output-directory "resources/tests/spark/export/"
                           :record-reader (eval (initialize-rr!
                                                 :rr (new-csv-record-reader)
                                                 :input-split fs
                                                 :as-code? false))
                           :batch-size 1
                           :regression? false
                           :label-idx 10
                           :n-labels 10)
                  :iter (-> (slurp "resources/poker-spark-test.csv")
                            (clojure.string/replace "\n" "")
                            vector
                            (java.util.ArrayList.)
                            .listIterator)))))
    ;; to test fns which require tuple inputs, datavec.spark packages need to be implemented
    #_(is (= "" (call-datavec-byte-ds-fn! :the-fn (new-datavec-byte-ds-fn :label-idx 10
                                                                          :n-labels 10
                                                                          :batch-size 1
                                                                          :byte-file-len poker-training-file-byte-size))))

    ;; implement the higher level implementation fns
    ;; see when we get back to the low level

    )9)
